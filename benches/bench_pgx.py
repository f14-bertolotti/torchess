import time
import jax
import pgx


def main():

    import jax
    import jax.numpy as jnp
    import pgx
    import time
    
    jax.config.update("jax_platform_name", "gpu")  # Ensure GPU usage
    
    env = pgx.make("chess")
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    
    batch_size = 4096
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
    
    print(f"Time taken: {end_time:.2f} seconds")
    print(f"Average time per step: {end_time/300:.5f} seconds")

if __name__ == '__main__':
    main()
