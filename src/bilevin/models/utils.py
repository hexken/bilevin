def update_common_params(args, params: dict):
    if args.optimizer == "SGD":
        params["momentum"] = args.momentum
        params["nesterov"] = args.nesterov
        params["weight_decay"] = args.weight_decay
