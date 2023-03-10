def print_args(args):
    
    print("")
    print("-" * 20, "args", "-" * 20)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 18, "args end", "-" * 18, flush=True)
