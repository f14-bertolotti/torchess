import jax
import jax.numpy as jnp
import pgx
import time
import click

def main(batch_size=4096):
   
    jax.config.update("jax_platform_name", "gpu")  # Ensure GPU usage
    
    env = pgx.make("chess")
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    actions = jax.random.randint(keys[0], (1000, batch_size), 0, 4672)
    
    # Convert actions to JAX array
    actions = jnp.array(actions)
    
    # Ensure state is a JAX array
    state = init(keys)

    # Force GPU warm-up (compilation overhead)
    state = step(state, actions[0])  # Run once before timing
    
    # Measure execution time
    start_time = time.time()
    for i in range(actions.shape[0]):
        state = step(state, actions[i])
    end_time = time.time() - start_time
    
    print(f"Time taken: {end_time} seconds")
    print(f"Average time per step: {end_time/actions.shape[0]} seconds")
    return end_time/actions.shape[0]

@click.command()
@click.option("--batch-size", default=4096, help="Batch size for parallel execution")
def cli(batch_size):
    main(batch_size)


if __name__ == '__main__':
    cli()
