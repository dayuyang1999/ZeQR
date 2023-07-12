from typing import Dict, List


def deduplicate_run(run_file: str) -> List:

    # {'106_1': [], '106_2' : [], ... }
    document_ids = {}

    with open(run_file) as f:

        run_rows: List = []

        for i, line in enumerate(f):

            line_dict: Dict = {}

            line_content = line.split()
            try:
                line_content[2] = line_content[2].rsplit("-",1)[0]
            except:
                print(f"Error happen in line {i}")
                raise ValueError

            # check if turn_id is in the dictionary
            if document_ids.get(line_content[0]):

                # check if turn has document id in its list
                if line_content[2] not in document_ids[line_content[0]]:

                    document_ids[line_content[0]].append(line_content[2])

                    line_dict = {
                        "turn_id": line_content[0],
                        "dummy_value": line_content[1],
                        "doc_id": line_content[2],
                        "rank": line_content[3],
                        "score": line_content[4],
                        "run_name": line_content[5]
                    }

                    run_rows.append(line_dict)
            else:

                document_ids[line_content[0]] = []
                document_ids[line_content[0]].append(line_content[2])

                line_dict = {
                    "turn_id": line_content[0],
                    "dummy_value": line_content[1],
                    "doc_id": line_content[2],
                    "rank": line_content[3],
                    "score": line_content[4],
                    "run_name": line_content[5]
                }

                run_rows.append(line_dict)

        return run_rows


def adjust_run_ranking(run_rows: List) -> List:

    for i in range(1, len(run_rows)):

        if run_rows[i]["turn_id"] == run_rows[i-1]["turn_id"]:
            run_rows[i]["rank"] = str(int(run_rows[i-1]["rank"]) + 1)

        else:
            run_rows[i]['rank'] = str(1)

    return run_rows


if __name__ == "__main__":
    run_file_dir = 'retrieval_results/ance_oriqincontext/2019.run'
    out_run_dir = '/'.join(run_file_dir.split('.')[0].split('/')[:-1])
    file_name  = run_file_dir.split('.')[0].split('/')[-1]
    print("out dir is:", out_run_dir)
    deduplicated_run = deduplicate_run(run_file_dir)
    with open(f"{out_run_dir}/{file_name}-dedup.run", "w") as cf:
        for row in deduplicated_run:
            cf.write(
                f'{row["turn_id"]} {row["dummy_value"]} {row["doc_id"]} {row["rank"]} {row["score"]} {row["run_name"]}\n')