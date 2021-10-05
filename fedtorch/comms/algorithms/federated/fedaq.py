# -*- coding: utf-8 -*-
import time
from copy import deepcopy

import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model
from fedtorch.comms.utils.flow_utils import (quantize_tensor,
                                             dequantize_tensor,
                                             compress_tensor,
                                             decompress_tensor)

def fedaq_aggregation(args, model_server, model_client, model_server_ag, model_client_ag,
                         group, online_clients, optimizer, lr, local_steps, lambda_weight=None):
    """Aggregate gradients for federated learning using FedAQ
    """
    sum_comm = 0.0
    st_comp = time.time()
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * args.graph.n_nodes / num_online_clients

    
    for server_param, client_param, server_ag_param, client_ag_param in zip(model_server.parameters(), model_client.parameters(), model_server_ag.parameters(), model_client_ag.parameters()):
        # get model difference.
        client_param.grad.data = (server_param.data - client_param.data) * rank_weight
        #client_ag_param.grad.data = (server_ag_param.data - client_ag_param.data) * rank_weight
        client_ag_param_grad = (server_ag_param.data - client_ag_param.data) * rank_weight
        if args.quantized:
            grad_q, q_info = quantize_tensor(client_param.grad.data, num_bits= args.quantized_bits, adaptive=True)
            gather_list_tensor = [torch.ones_like(grad_q) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            gather_list_info   = [torch.ones(3) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            grad_q_ag, q_info_ag = quantize_tensor(client_ag_param_grad, num_bits= args.quantized_bits, adaptive=True)
            gather_list_tensor_ag = [torch.ones_like(grad_q_ag) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            gather_list_info_ag   = [torch.ones(3) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            
            st = time.time()
            dist.gather(q_info, gather_list=gather_list_info, dst=0, group=group)
            dist.gather(grad_q, gather_list=gather_list_tensor, dst=0, group=group)
            dist.gather(q_info_ag, gather_list=gather_list_info_ag, dst=0, group=group)
            dist.gather(grad_q_ag, gather_list=gather_list_tensor_ag, dst=0, group=group)
            #args.comm_time[-1] += time.time() - st
            sum_comm += time.time() - st

            if args.graph.rank == 0:
                gather_list_tensor = gather_list_tensor if 0 in online_clients else gather_list_tensor[1:]
                gather_list_info = gather_list_info if 0 in online_clients else gather_list_info[1:]
                gather_list_deq = [dequantize_tensor(t,i) for t,i in zip(gather_list_tensor,gather_list_info)]
                gather_list_tensor_ag = gather_list_tensor_ag if 0 in online_clients else gather_list_tensor_ag[1:]
                gather_list_info_ag = gather_list_info_ag if 0 in online_clients else gather_list_info_ag[1:]
                gather_list_deq_ag = [dequantize_tensor(t,i) for t,i in zip(gather_list_tensor_ag,gather_list_info_ag)]

                d = torch.sum(torch.stack(gather_list_deq,1), dim=1)
                d, avg_info = quantize_tensor(d,num_bits= args.quantized_bits, adaptive=True)
                d_ag = torch.sum(torch.stack(gather_list_deq_ag,1), dim=1)
                d_ag, avg_info_ag = quantize_tensor(d_ag,num_bits= args.quantized_bits, adaptive=True)
            else:
                d = torch.ones_like(grad_q)
                avg_info = torch.ones(3)
                d_ag = torch.ones_like(grad_q_ag)
                avg_info_ag = torch.ones(3)
            
            st = time.time()
            dist.broadcast(avg_info, src=0, group=group)
            dist.broadcast(d, src=0, group=group)
            dist.broadcast(avg_info_ag, src=0, group=group)
            dist.broadcast(d_ag, src=0, group=group)
            #args.comm_time[-1] += time.time() - st
            sum_comm += time.time() - st
            client_param.grad.data = dequantize_tensor(d,avg_info)
            client_ag_param_grad = dequantize_tensor(d_ag, avg_info_ag)
        else:
            # all reduce. This schema is not used in real federated setting
            # dist.all_reduce(client_param.grad.data,  op=dist.ReduceOp.SUM, group=group)

            #### Federated communication #####
            gather_list = [torch.ones_like(client_param.grad.data) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            gather_list_ag = [torch.ones_like(client_ag_param_grad) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            st = time.time()
            dist.gather(client_param.grad.data, gather_list=gather_list, dst=0, group=group)
            dist.gather(client_ag_param_grad, gather_list=gather_list_ag, dst=0, group=group)
            #args.comm_time[-1] += time.time() - st
            sum_comm += time.time() - st
            if args.graph.rank == 0:
                gather_list = gather_list if 0 in online_clients else gather_list[1:]
                d = torch.sum(torch.stack(gather_list,1), dim=1)
                gather_list_ag = gather_list_ag if 0 in online_clients else gather_list_ag[1:]
                d_ag = torch.sum(torch.stack(gather_list_ag,1), dim=1)
            else:
                d = torch.ones_like(client_param.grad.data)
                d_ag = torch.ones_like(client_ag_param_grad)
            st = time.time()
            dist.broadcast(d, src=0, group=group)
            dist.broadcast(d_ag, src=0, group=group)
            #args.comm_time[-1] += time.time() - st
            sum_comm += time.time() - st
            client_param.grad.data = d
            client_ag_param_grad = d_ag
    
        #get two types of parameters after synchronization
        client_param.data = server_param.data - client_param.grad.data
        client_ag_param.data = server_ag_param.data - client_ag_param_grad

    # apply gradient to each client's model
    #optimizer.step(
    #    apply_lr=False,
    #    scale=args.lr_scale_at_sync,
    #    apply_in_momentum=False,
    #    apply_out_momentum=args.out_momentum,
    #)
    #for client_param, client_ag_param in zip(model_client.parameters(), model_client_ag.parameters()):
    #    client_param.data -= client_param.grad.data
    #    client_ag_param.data -= client_ag_param_grad
    args.comp_time[-1] += (time.time() - st_comp - sum_comm) 
    args.comm_time[-1] += sum_comm

    # Reassign model_client to model_server
    model_server = deepcopy_model(args, model_client)
    model_server_ag = deepcopy_model(args, model_client_ag)

    return model_server, model_server_ag


def distribute_model_server_aq(model_server, model_server_ag, group, src=0):
    """
    Distributing the model on server from source node to the group of process
    #TODO: merge with distribute_model_server method
    """
    for server_param, server_ag_param in zip(model_server.parameters(),model_server_ag.parameters()):
        t = torch.stack([server_param.data, server_ag_param.data])
        dist.broadcast(t, src=src, group=group)
        server_param.data = t[0]
        server_ag_param.data = t[1]


    return model_server, model_server_ag
