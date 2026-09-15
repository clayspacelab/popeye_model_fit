# import argparse

# def parse_force_cpu():
#     parser = argparse.ArgumentParser(add_help=False)
#     parser.add_argument("--force-cpu", action="store_true")
#     args, _ = parser.parse_known_args()
#     return args.force_cpu

def configure_jax(force_cpu=False):
    import jax
    if force_cpu:
        jax.config.update("jax_platform_name", "cpu")

    return jax.default_backend()