import argparse
import os
import socket
import time
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ctx_servers", type=int, required=True, help="Number of context servers")
    parser.add_argument("--num_gen_servers", type=int, required=True, help="Number of generation servers")
    parser.add_argument("--work_dir", type=str, default="logs", help="Work directory")
    parser.add_argument("--worker_port", type=int, default=8336, help="Worker port")
    parser.add_argument("--server_port", type=int, default=8300, help="Base server port (for first coordinator)")
    parser.add_argument("--num_server_instances", type=int, default=1, help="Number of coordinator server instances")
    parser.add_argument("--server_postprocess_workers", type=int, default=4, help="Number of postprocess workers for coordinator")
    parser.add_argument("--server_workers_per_core", type=int, default=2, help="Worker threads per CPU core for coordinator")
    args = parser.parse_args()

    # Check if the work_dir exists
    if not os.path.exists(args.work_dir):
        raise ValueError(f"Work directory {args.work_dir} not found")

    # Wait for all hostname files to be created
    hostnames_folder = os.path.join(args.work_dir, "hostnames")
    while not os.path.exists(hostnames_folder):
        time.sleep(10)
    
    hostnames = os.listdir(hostnames_folder)
    expected_count = args.num_ctx_servers + args.num_gen_servers
    while len(hostnames) != expected_count:
        time.sleep(10)
        hostnames = os.listdir(hostnames_folder)

    # Read CTX and GEN URLs
    ctx_urls = []
    gen_urls = []
    for hostname_file in sorted(hostnames):
        hostname_file_path = os.path.join(hostnames_folder, hostname_file)
        with open(hostname_file_path, 'r') as f:
            url = f.read().strip()

        if hostname_file.startswith("CTX"):
            ctx_urls.append(url)
        elif hostname_file.startswith("GEN"):
            gen_urls.append(url)

    # Get current hostname
    hostname = socket.gethostname()

    # Generate server config(s) - one for each coordinator instance
    server_urls = []
    for server_idx in range(args.num_server_instances):
        server_port = args.server_port + server_idx
        server_url = f"{hostname}:{server_port}"
        server_urls.append(server_url)
        
        # Single coordinator: use actual hostname (simple, works like before)
        # Multi-coordinator: use 0.0.0.0 to allow running on any node
        if args.num_server_instances == 1:
            bind_hostname = hostname  # Simple case - bind to this specific node
        else:
            bind_hostname = '0.0.0.0'  # Distributed - bind to all interfaces
        
        server_config = {
            'hostname': bind_hostname,
            'port': server_port,
            'backend': 'pytorch',
            'num_postprocess_workers': args.server_postprocess_workers,
            'workers_per_core': args.server_workers_per_core,
            'context_servers': {
                'num_instances': args.num_ctx_servers,
                'urls': ctx_urls
            },
            'generation_servers': {
                'num_instances': args.num_gen_servers,
                'urls': gen_urls
            }
        }

        if args.num_server_instances == 1:
            # Backward compatibility: single file named server_config.yaml
            config_file = os.path.join(args.work_dir, "server_config.yaml")
        else:
            # Multiple instances: server_config_0.yaml, server_config_1.yaml, etc.
            config_file = os.path.join(args.work_dir, f"server_config_{server_idx}.yaml")
        
        with open(config_file, "w") as f:
            yaml.dump(server_config, f, default_flow_style=False, sort_keys=False)
    
    # Write server URLs file for harness to read
    server_urls_file = os.path.join(args.work_dir, "server_urls.txt")
    with open(server_urls_file, "w") as f:
        f.write(",".join(server_urls))

