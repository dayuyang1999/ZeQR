import os
import gzip
import argparse
from dedup import deduplicate_run, adjust_run_ranking

parser = argparse.ArgumentParser(
    description='Convert CAsT Passage Run Files to Document Run Files')

parser.add_argument('--run_file_path', type=str, help="Path to run file")
parser.add_argument('--base_path', type=str, help="base path to ZeQR directory. For example: /usa/dayu/ZeQR/")
parser.add_argument('--run_name', type=str, help="name of run")

args = parser.parse_args()


with open(args.run_file_path, "rb") as run_file:

    intermediary = run_file.read()

    file_basename = args.run_name
    
    #os.path.splitext(os.path.basename(args.run_file_path))[0]

    with open(f"{file_basename}.run", "wb") as f:

        f.write(intermediary)


deduplicated_run = deduplicate_run(f"{file_basename}.run")
adjusted_run = adjust_run_ranking(deduplicated_run)



with open(os.path.join(args.base_path, f"ranking_results/{file_basename}-converted.run"), "w") as cf:

    for row in adjusted_run:
        cf.write(
            f'{row["turn_id"]} {row["dummy_value"]} {row["doc_id"]} {row["rank"]} {row["score"]} {row["run_name"]}\n')