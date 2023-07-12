from tqdm import tqdm
import os
import pickle
import json
import argparse



class RunFuser():
    '''
    output(method run) is store 1 directory above the turn run
    
    '''
    def __init__(self, output_dir:str, run_name:str, year:str, combined_run_name:str, index_shards:int):
        self.concat_num =2000
        self.index_shards = 20
        self.output_dir = output_dir
        self.run_name = run_name
        self.year = year
        self.combined_run_name = combined_run_name
        self.raw_query_field = 'raw_utterance'

        
        if self.year == 2020:
            self.topic_dir = "CAsT/topics/2020_manual_evaluation_topics_v1.0.json"
            self.qrel_file_dir = "CAsT/qrels/2020qrels.txt"
            raw_query_field = 'raw_utterance'
        elif self.year == 2019:
            self.topic_dir = 'CAsT/topics/2019_evaluation_topics_v1.0.json'
            self.qrel_file_dir = 'CAsT/qrels/2019qrels.txt'
            raw_query_field = 'raw_utterance'
        elif self.year == 2021:
            self.topic_dir = 'CAsT/topics/2021_manual_evaluation_topics_v1.0.json'
            self.qrel_file_dir = 'CAsT/qrels/2021qrels.txt'
            raw_query_field = 'raw_utterance'
        elif self.year == 2022:
            self.topic_dir = "CAsT/topics/2022_evaluation_topics_flattened_duplicated_v1.0.json"
            self.qrel_file_dir = 'CAsT/qrels/2022qrels.txt'
            raw_query_field = 'utterance'
        else:
            raise ValueError("year not support")
        
        
        self.qid_lst = self.create_qid_lst()
        
        self.combine_dense()


    def create_qid_lst(self):
        # create all qid list
        all_qids = []
        with open(self.topic_dir, 'r') as topics:
            topics = json.load(topics)
        for session in topics:
            session_num = str(session["number"])
            for turn_id, conversations in enumerate(session["turn"]):
                #query = conversations[self.raw_query_field]
                conversation_num = str(conversations["number"])
                qid = session_num + "_" + conversation_num
                if qid in all_qids:
                    pass
                else:
                    all_qids.append(qid)
        all_qids = list(set(all_qids))
        print(f"totally {len(all_qids)} number of qids")
        return all_qids
        
    

    def fusing_indeces(self, qid:str, concat_num:int):
        '''
        fusing all indeces under the same qid using retrieval score
        
        '''
        all_hits = []
        for index_id in range(self.index_shards):
            with open(os.path.join(self.output_dir, self.run_name, self.year, f'dense{index_id}-{qid}.pkl'), 'rb') as focal_hits:
                focal_hits = pickle.load(focal_hits)
            all_hits += focal_hits
        scores = [hit.score for hit in all_hits]
        reranking = list(zip(all_hits, scores))
        reranking.sort(key=lambda x: x[1], reverse=True)
        reranked_hits = []
        for hit, score in reranking:
            hit.score = score
            reranked_hits.append(hit)
        
        return reranked_hits[:concat_num]



    
    #@dask.delayed
    def hits_to_runfile(self, hits, qid:str) -> None:
        '''
        q_level hits to q_level runfile
        '''
        out_file = os.path.join(self.output_dir, self.run_name, self.year, f'dense-{qid}.run')
        with open(out_file, 'w') as fout:
            for rank in range(len(hits)):
                if self.year in ['2021', '2022']:
                    docno = hits[rank].docid
                elif self.year in ['2019', '2020']:
                    docno = hits[rank].docid.split('-')[0]
                else:
                    raise ValueError("year not support")
                score = hits[rank].score
                fout.write("{} Q0 {} {} {} {}\n".format(qid, docno, rank + 1, score, self.run_name)) 


    def fusing_method(self, all_qid:list):
        '''
        fusing all qid under the same method
        
        '''
        out_file_dir = os.path.join(self.output_dir, self.run_name, f'{self.combined_run_name}.run')
        with open(out_file_dir, 'w') as fout:
            for qid in self.qid_lst:
                try:
                    q_level_infile_dir = os.path.join(self.output_dir, self.run_name, self.year, f'dense-{qid}.run')
                    with open(q_level_infile_dir, 'r') as fin:
                        fout.write(fin.read())
                except:
                    pass
                    print(f"WARNING: The method-level retrieval result of {qid} not found, skip")
    
    def combine_dense(self):
        '''
        Combine all dense runs from multiple indeces

        '''
        for qid in tqdm(self.qid_lst):
            #self.hits_to_runfile(self.fusing_indeces(qid, self.concat_num), qid)
            try:
                self.hits_to_runfile(self.fusing_indeces(qid, self.concat_num), qid)
            except:
                pass
                print(f"WARNING: The sub-index level retrieval result of {qid} not found, skip")
        
        print(f"Combing all qid into: {self.combined_run_name}.run")
        
        self.fusing_method(self.qid_lst)

        print('Done Combination')




class Evaluator():
    def __init__(self, year:int, eval_top=1000):
        self.year = year
        self.topic_dir, self.qrel_file_dir, self.raw_query_field = self.load_cast_files()
        self.all_qids = self.create_qid_list()
        self.eval_top = eval_top
        
    
    
    
    
    def load_cast_files(self):

        if self.year == 2020:
            topic_dir = "CAsT/topics/2020_manual_evaluation_topics_v1.0.json"
            qrel_file_dir = "CAsT/qrels/2020qrels.txt"
            raw_query_field = 'raw_utterance'
        elif self.year == 2019:
            topic_dir = 'CAsT/topics/2019_evaluation_topics_v1.0.json'
            qrel_file_dir = 'CAsT/qrels/2019qrels.txt'
            raw_query_field = 'raw_utterance'
        elif self.year == 2021:
            topic_dir = 'CAsT/topics/2021_manual_evaluation_topics_v1.0.json'
            qrel_file_dir = 'CAsT/qrels/2021qrels.txt'
            raw_query_field = 'raw_utterance'
        elif self.year == 2022:
            topic_dir = "CAsT/topics/2022_evaluation_topics_flattened_duplicated_v1.0.json"
            qrel_file_dir = 'CAsT/qrels/2022qrels.txt'
            raw_query_field = 'utterance'
        else:
            raise ValueError("year not support")
        return topic_dir, qrel_file_dir, raw_query_field
    
    def create_qid_list(self):
        all_qids = []
        with open(self.topic_dir, 'r') as topics:
            topics = json.load(topics)
        for session in topics:
            session_num = str(session["number"])
            for turn_id, conversations in enumerate(session["turn"]):
                #query = conversations[raw_query_field]
                conversation_num = str(conversations["number"])
                qid = session_num + "_" + conversation_num
                if qid in all_qids:
                    pass
                else:
                    all_qids.append(qid)

        all_qids = list(set(all_qids))

        print(f"totally {len(all_qids)} number of qids")
        
        return all_qids
            

        
    def eval2021(self, hits_num, run_file_dir:str, run_name:str, qrel_file_dir:str)-> None:
        '''
        Only 2021 need to docize the runfile
        '''
        docize_file_dir = "/data_hdd/dayu/TREC/run_results/official_docize.py"
        
        
        print("evaluating...")
        os.system(f'python3 {docize_file_dir} --run_file_path {run_file_dir} --run_name {run_name}')
        
        # os.system(f'python -m pyserini.eval.trec_eval -c -M 1000 -m map --level_for_rel 2 {qrel_file_dir} /data_hdd/dayu/TREC/run_results/{run_name}-converted.run')
        # os.system(f'python -m pyserini.eval.trec_eval -c -M 1000 -m mrr --level_for_rel 2 {qrel_file_dir} /data_hdd/dayu/TREC/run_results/{run_name}-converted.run')
        # os.system(f'python -m pyserini.eval.trec_eval -c -M 1000 -m recall --level_for_rel 2 {qrel_file_dir} /data_hdd/dayu/TREC/run_results/{run_name}-converted.run')
        # os.system(f'python -m pyserini.eval.trec_eval -c -M {hits_num} -m recall.{hits_num} --level_for_rel 2 {qrel_file_dir} /data_hdd/dayu/TREC/run_results/{run_name}-converted.run')
        # os.system(f'python -m pyserini.eval.trec_eval -c -M 1000 -m ndcg_cut.3 {qrel_file_dir} /data_hdd/dayu/TREC/run_results/{run_name}-converted.run')
        # os.system(f'python -m pyserini.eval.trec_eval -c -M {hits_num} -m all_trec {qrel_file_dir} /data_hdd/dayu/TREC/run_results/{run_name}-converted.run')
        os.system(f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.3 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.5 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m P.5 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m recall.100 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m map {qrel_file_dir} {run_file_dir}')
        
        print("Evaluation Done")
        
        
    def eval(self, hits_num, run_file_dir:str, qrel_file_dir:str)-> None:
        '''
        
        '''
        #docize_file_dir = "/data_hdd/dayu/TREC/run_results/official_docize.py"
        
        
        print("evaluating...")
        #os.system(f'python3 {docize_file_dir} --run_file_path {run_file_dir} --run_name {run_name}')
        
        # os.system(f'python -m pyserini.eval.trec_eval -c -m map_cut.100 --level_for_rel 2 {qrel_file_dir} {run_file_dir}')
        # os.system(f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.3 {qrel_file_dir} {run_file_dir}')
        # os.system(f'python -m pyserini.eval.trec_eval -c -M {hits_num} -m all_trec {qrel_file_dir} {run_file_dir}')

        os.system(f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.3 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.5 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m P.5 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m recall.100 {qrel_file_dir} {run_file_dir}')
        os.system(f'python -m pyserini.eval.trec_eval -c -m map {qrel_file_dir} {run_file_dir}')
        
        print("Evaluation Done")

        
    
    
    
    def evaluate(self, run_name:str, year:int, output_dir='./ranking_results/',combined_run_name ='dense', shards=20):
        _ = RunFuser(output_dir=output_dir, run_name=run_name, year=str(year), combined_run_name=combined_run_name, index_shards=shards)
        combined_run_dir = os.path.join(output_dir, run_name, f'{combined_run_name}.run')
        if self.year in [2021, 2019]: # 2019 for dedup
            self.eval2021(self.eval_top, combined_run_dir, run_name, self.qrel_file_dir)
        else:
            self.eval(self.eval_top, combined_run_dir, self.qrel_file_dir)
        

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the run file')
    parser.add_argument('--year', type=int, default=2019, help='year of the run file')
    parser.add_argument('--run_name', type=str, default='not_provided', help='name of the run file')
    parser.add_argument('--index_shards', type=int, default=20, help='number of shards of the index')
    return parser.parse_args()      
            
if __name__ == '__main__':
    args = parse_args()
    # run evaluation
    evaluator = Evaluator(args.year)
    if args.run_name == 'not_provided':
        evaluator.evaluate(run_name="CAsT"+str(args.year), year=args.year, combined_run_name=str(args.year), shards=args.index_shards)
    else:
        evaluator.evaluate(run_name=args.run_name, year=args.year, combined_run_name=args.run_name, shards=args.index_shards)
    