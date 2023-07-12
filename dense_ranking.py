from tqdm import tqdm
from pyserini.dsearch import SimpleDenseSearcher
from pyserini.search import JSimpleSearcherResult
import argparse
import json
import pickle
import pandas as pd
from typing import List, Dict
from MRC_query_reformulator import MRC_Query_Reformulator
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Dense retrieval experiments for TREC CAsT.')
    parser.add_argument('--run_name', type=str, default='test', required=True, help='run file name printed in trec file')
    parser.add_argument('--year', type=str, default='2021', choices=['2019', '2020', '2021', '2022'], help='year of TREC CAsT')
    
    
    ###### overall settings
    parser.add_argument('--verbose', type=bool, default=False, help='verbose')
    parser.add_argument('--device', type=int, default='1', help='device to use')
    parser.add_argument('--query_file_dir', type=str, required=True, help='query file with required format, see README.md for more info')
    parser.add_argument('--index_base_dir', default='index/', type=str, required=False, help='base index dir')
    parser.add_argument('--output_dir', type=str, default='retrieval_results/', help='output dir')
    
    ###### search settings
    parser.add_argument('--hits', default=1000, help='number of hits to retrieve')
    parser.add_argument('--query_encoder', default= 'castorini/ance-msmarco-passage') # castorini/ance-msmarco-passage
    
    ###### experiment settings
    
    ###### reformulator settings
    parser.add_argument('--reformulator', type=str, default='mrc', choices=['no', 't5', 'mrc', 'human', 'transformer', 'convdrzs'], help='reformulator name')
    parser.add_argument('--transformer_rewritten_queries', type=str, default='data/transformer_rewritten_queries.json', help='transformer rewritten queries')
    
    
    # MRC reformulator settings
    parser.add_argument('--search_in_qa_topk', type=int, default=5, help='only search in qa top k')
    parser.add_argument('--turnoffcoref_module_activate', action='store_true', help='turnoff coref module')
    parser.add_argument('--turnoffomission_module_activate', action='store_true', help='turnof coref module')
    parser.add_argument('--add_oriq_in_context', action='store_true', help='append original question qn at the end of the context')
    parser.add_argument('--add_oriq_in_question', action='store_true', help='append original question qn at the end of the question')
                        #type=bool, default=True, help='add original question in context')
    parser.add_argument('--max_top_k', type=int, default=2, help='max top k important words to find ambiguity')
    parser.add_argument('--max_ambiguity_type_loop', type=int, default=3, help='max ambiguity type loop')
    parser.add_argument('--answer_max_length', type=int, default=5, help='max length of answer')
    parser.add_argument('--minimum_bm25_score', type=float, default=2.65, help='minimum bm25 score')
    parser.add_argument('--reformulator_device', type=int, default=1, help='device to use')
    parser.add_argument('--sort_by_length', type=bool, default=False, help='sort by length')
    parser.add_argument('--reformulator_verbose', type=bool, default=False, help='verbose')
    parser.add_argument('--ref_model', type=str, default='huggingface-course/bert-finetuned-squad', help='ref model')
    parser.add_argument('--des_model', type=str, default='huggingface-course/bert-finetuned-squad', help='des model')
    parser.add_argument('--mark', type=str, default='"', help='mark to identify the focal noun/verb/pronoun')
    
    # Coref reformulator settings
    parser.add_argument('--coref_rewritten_out_dir', type=str, default="coref/trec_rewritten", help='coref model result output dir')
    
    return parser.parse_args()


def build_reformulator(reformulator_name:str, args):
    '''
    Purpose:
        - build reformulator
            - which take: 
                - original question
                - context
            - and output:
                - reformulated question
        
    
    Output:
        - reformulator
    '''
    if reformulator_name == 'mrc':
        coref_module_activate = (not args.turnoffcoref_module_activate)
        omission_module_activate = (not args.turnoffomission_module_activate)
        print("coref of mrc status:", coref_module_activate)
        print("omission of mrc status:", omission_module_activate)
        reformulator = MRC_Query_Reformulator(
                                                  max_topk=args.max_top_k, 
                                                  max_ambiguity_type_loop=args.max_ambiguity_type_loop, 
                                                  answer_max_length = args.answer_max_length,
                                                  minimum_bm25_score=args.minimum_bm25_score, 
                                                  device=args.reformulator_device,
                                                  sort_by_length=args.sort_by_length, 
                                                  verbose=args.reformulator_verbose, 
                                                  ref_model=args.ref_model, 
                                                  des_model=args.des_model, 
                                                  add_oriq_in_context = args.add_oriq_in_context,
                                                  add_oriq_in_question = args.add_oriq_in_question,
                                                  search_in_qa_topk=args.search_in_qa_topk,
                                                  coref_module_activate=coref_module_activate,
                                                  omission_module_activate=omission_module_activate,
                                                  mark = args.mark
                                                  )
    elif reformulator_name == 'coref':
        with open(os.path.join(args.coref_rewritten_out_dir, args.year, 'rewritten_qs.json'), 'r') as f:
            reformulator = json.load(f)
    elif reformulator_name == "coref_mrc":
        coref_module_activate = (not args.turnoffcoref_module_activate)
        omission_module_activate = (not args.turnoffomission_module_activate)
        print("coref of mrc status:", coref_module_activate)
        print("omission of mrc status:", omission_module_activate)
        reformulator_mrc = MRC_Query_Reformulator(
                                                  max_topk=args.max_top_k, 
                                                  max_ambiguity_type_loop=args.max_ambiguity_type_loop, 
                                                  answer_max_length = args.answer_max_length,
                                                  minimum_bm25_score=args.minimum_bm25_score, 
                                                  device=args.reformulator_device,
                                                  sort_by_length=args.sort_by_length, 
                                                  verbose=args.reformulator_verbose, 
                                                  ref_model=args.ref_model, 
                                                  des_model=args.des_model, 
                                                  add_oriq_in_context = args.add_oriq_in_context,
                                                  add_oriq_in_question = args.add_oriq_in_question,
                                                  search_in_qa_topk=args.search_in_qa_topk,
                                                  coref_module_activate=coref_module_activate,
                                                  omission_module_activate=omission_module_activate,
                                                  mark=args.mark
                                                  )
        with open(os.path.join(args.coref_rewritten_out_dir, args.year, 'rewritten_qs.json'), 'r') as f:
            reformulator_coref = json.load(f)
        reformulator = {'mrc': reformulator_mrc, 'coref': reformulator_coref}
    elif (reformulator_name == 'human') or (reformulator_name == 'no'):
        reformulator = None
    elif reformulator_name == 'convdrzs':
        with open(f"CAsT/query_file/convdrzs_query_{args.year}.json", 'r') as f:
            reformulator = json.load(f)
    
    elif reformulator_name == 'transformer':
        with open(args.transformer_rewritten_queries, 'r') as f:
            reformulator = json.load(f)
    
    return reformulator
        



def first_dense_rank(index_i, query_file: List[Dict], reformulator, dense_searcher, args):
    '''
    Purpose:
        - rank the first dense retrieval results
        - output the trec file
        
    Input:
        - query_file
        - reformulator
        - dense_searcher
        - args
    '''
    
    if not os.path.exists(os.path.join(args.output_dir, args.run_name, args.year)):
        os.makedirs(os.path.join(args.output_dir, args.run_name, args.year))
    
    for turn in tqdm(query_file):
        qid = turn['qid']
        query = turn['ori_q']
        context = turn['context']
        
        # reformulate the query
        if args.reformulator == 'no':
            reformulated_query = query
        elif args.reformulator == 'mrc':
            reformulated_query = reformulator.reformulate(query, context)
        elif args.reformulator == 'coref':
            reformulated_query = reformulator[qid] # you can rewrite on time if you want
        elif args.reformulator == 'coref_mrc':
            reformulated_query = reformulator['mrc'].reformulate(reformulator['coref'][qid], context)
        elif args.reformulator == 'human':
            reformulated_query = turn['rew_q']
        elif args.reformulator == 'transformer':
            reformulated_query = reformulator[qid]
        elif args.reformulator == 'convdrzs':
            reformulated_query = reformulator[qid]
        else:
            raise ValueError(f'args.reformulator: {args.reformulator} is not supported')
        
        if args.verbose:
            print(f'qid: {qid}, query: {query}, reformulated_query: {reformulated_query}')
        
        # retrieve the top 2000 dense results
        hits = dense_searcher.search(reformulated_query, k=args.hits)
        
        # write the trec file
        # out_file = os.path.join(args.output_dir, args.run_name, args.year, f'dense_{index_i}.run')
        # with open(os.path.join(args.output_dir, args.run_name, args.year, f'dense_{index_i}.trec'), 'w+') as f:
        #     for rank, hit in enumerate(hits):
        #         f.write(f'{qid} Q0 {hit.docid} {rank+1} {hit.score} {args.run_name}')
        
        # directly store the hits
        out_file_dir = os.path.join(args.output_dir, args.run_name, args.year, f'dense{index_i}-{qid}.pkl')
        with open(out_file_dir, 'wb') as f:
            pickle.dump(hits, f)
  
  
    
def hits_to_runfile(self, hits:List[JSimpleSearcherResult], out_file:str, qid:str ) -> None:

    with open(out_file, 'w') as fout:
        for rank in range(len(hits)):
            docno = hits[rank].docid
            score = hits[rank].score
            fout.write("{} Q0 {} {} {} {}\n".format(qid, docno, rank + 1, score, self.run_name)) 
    

    


if __name__ == '__main__':
    
    args = parse_args()
    print("reformulater using:", args.reformulator)

    
    # load query_file
    with open(args.query_file_dir, 'r') as f:
        query_file = json.load(f)

    
    
    # build reformulator
    
    reformulator = build_reformulator(args.reformulator, args)
    
    # search
    for index_i in tqdm(range(20)): # loop over 20 indexes
        dense_searcher = SimpleDenseSearcher(os.path.join(args.index_base_dir,f'dense_{args.year}', f'cast{args.year}_den{index_i}'), args.query_encoder)
        first_dense_rank(index_i, query_file, reformulator, dense_searcher, args)
