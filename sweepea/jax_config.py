#implement force_cpu functionality
def configure_jax(force_cpu=False):
    import jax
    if force_cpu:
        jax.config.update("jax_platform_name", "cpu")

    return jax.default_backend()