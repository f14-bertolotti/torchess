import json
import click

from benches.bench_pgx import main as pgx
from benches.bench_pwn import main as pwn

@click.command()
@click.option("--output-path", default="data.jsonl", help="Path to save benchmark results")
def benchmark(output_path):
    with open(output_path, "w") as f:
        for i in [2**x for x in range(1,13)]:
            print(f"Batch size: {i}")
            pgx_time = pgx(i)
            pwn_time = pwn(i)
            print(pgx_time, pwn_time, pgx_time/pwn_time)
            f.write(json.dumps({"batch_size": i, "pgx_time": pgx_time, "pwn_time": pwn_time}) + "\n")
    

if __name__ == '__main__':
    benchmark()

