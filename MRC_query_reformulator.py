#from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import TokenClassificationPipeline
from pyserini.search import SimpleSearcher
from transformers import pipeline
from typing import List
import string


class MRC_Query_Reformulator():
    def __init__(self, coref_module_activate=True, max_topk=1, max_ambiguity_type_loop=3, add_oriq_in_context = False, add_oriq_in_question=False, answer_max_length = 5, minimum_bm25_score=2.65, search_in_qa_topk=5, device=0, untrained_model = 'huggingface-course/bert-finetuned-squad', ref_model = 'huggingface-course/bert-finetuned-squad', des_model = 'huggingface-course/bert-finetuned-squad', pos_model= "vblagoje/bert-english-uncased-finetuned-pos",  sort_by_length=False, verbose=True, mark='"', omission_module_activate=True):
        '''
        Purpose:
        
        Edit: 2023-02-06
            - Edit: top 5 samples, if top1 is replicate with existing samples, then use top2...until top5
        
        Inputs:
            - max_topk: max top k important words to find ambiguity
                - for example, if max_topk = 1, then only find the 1st important word (seperated for noun and verb)
                
            - DEPRECIATED: max_ambiguity_type_loop: max loop to find ambiguity type
            - search_in_qa_topk: search in top k results in QA pipeline
            - device: device to use
                - -1 means cpu
                - 0 means gpu number 0
                - 1 ...
            - qa_model: qa model to use
            - pos_model: pos model to use
            - sparse_index_path: path to sparse index
        
        
        '''
        self.verbose = verbose
        self.device = device
        
        ##### if coref (QA version of coref) is used
        self.coref_module_activate = coref_module_activate
        self.omission_module_activate = omission_module_activate
        
        ##### answer filter settings
        self.add_oriq_in_context = add_oriq_in_context
        self.add_oriq_in_question = add_oriq_in_question
        self.answer_max_length = answer_max_length # cannot be longer than ? words (not characters)
        self.minimum_bm25_score = minimum_bm25_score
        
        # create pos model
        self.pos_model = AutoModelForTokenClassification.from_pretrained(pos_model)
        self.pos_tokenizer = AutoTokenizer.from_pretrained(pos_model)
        self.pos_pipl = TokenClassificationPipeline(model=self.pos_model, tokenizer=self.pos_tokenizer, device=self.device)
        
        # create qa model
        self.ref_pipl = pipeline("question-answering", model=ref_model, tokenizer=untrained_model, device=self.device)
        self.des_pipl = pipeline("question-answering", model=des_model, tokenizer=untrained_model, device=self.device)
        self.sort_by_length = sort_by_length
        
        
        # create sparse search model
        self.searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage') # change to use prebuilt index
        
        ## NOUN settings
        
        # pronouns
        people_pronouns1 = ['it', 'he','she', 'they'] # not use you
        people_pronouns2 = ['him', 'them']
        people_pronouns3 = ['his', 'their', 'her']
        people_pronouns4 = ['its', 'hers','theirs'] # newly added its
        thing_pronouns = ['one', 'ones', 'this', 'that', 'those', 'these', 'first','second', 'third', 'last', 'both']
        
        self.overallpronouns = people_pronouns1 + people_pronouns2 +people_pronouns3+ people_pronouns4 + thing_pronouns #+ ask_pronouns
        self.peoplesentity = people_pronouns3
        # nouns
        self.possible_important_nouns = ['NOUN', 'PRON']
        
        # ambiguity settings
        self.mark = mark
        
        # question settings
        self.question_marks = ['What', 'Does', 'How', 'Can', '']
        
        # find important words setting
        self.max_topk=max_topk # must >=1 (1 = the 1st important word
        assert self.max_topk >= 1, 'max_topk must >=1'
        self.max_ambiguity_type_loop = max_ambiguity_type_loop # max loop to find ambiguity type
        self.search_in_qa_topk = search_in_qa_topk # search in top k results in QA pipeline
        self.max_des_noun_ambiguity_loop = 2
        self.max_des_verb_ambiguity_loop = 2
    
    
    
    def mark_ambiguity_word(self, ori_utterance:str, ambiguity_word:str) -> str:
        '''
        mark the ambiguity word in the utterance to give prompt to the language model about
            where the ambiguity word locates
        
        '''
        split_pieces = ori_utterance.split(ambiguity_word)
        marked_utterance = f'{self.mark}{ambiguity_word}{self.mark}'.join(split_pieces)
        return marked_utterance

    
    
    def reformulate(self, query:str, context:str) -> str:
        '''
        Purpose:
            - given raw query, context, reformulate the query
            - Rule:
                - 1 step: clear all reference ambiguity
                - 2 step: clear ONLY-ONCE 2 types of des ambiguity
                    - 2.1 step: clear des_noun ambiguity max = ONLY-ONCE
                    - 2.2 step: clear des_verb ambiguity max = ONLY-ONCE
            
        
        
        '''
        if context == '':
            return query
        else:
            # if self.add_oriq_in_context:
            #     marked_query = self.mark_ambiguity_word(query)
            #     context = context + ' ' + marked_query
            # add some space to the utterance to make it easier to find ambiguity

            utterance_ = self.clean_raw_utterance(query)
            # ## loop over ambiguity type
            # for i in range(self.max_ambiguity_type_loop):
            #     ambiguity_type, ambiguity_list = self.find_ambiguity(utterance_)
            #     # loop over ambiguity in a ambiguity type: e.g. using 2 pronouns in a sentence
            #     if len(ambiguity_list) != 0: # by the definition of "find ambiguity" function, if len(ambiguity_list) == 0, then it means there is no ambiguity left in the utterance
            #         for ambiguity_info in ambiguity_list:
            #             answer = self.find_qa_results(ambiguity_type=ambiguity_type, ambiguity_info=ambiguity_info, raw_utterance=utterance_, context=context, question_answer_pip=self.qa_pipl)
            #             new_utterance = self.rewriting(answer, ambiguity_type, ambiguity_info, utterance_)
            #             utterance_ = new_utterance # update utteracne
                
            #     else:
            #         break
            
            # Rule: do not find ambiguity on previously added information
            
            
            ## 1st find pronoun ambiguity
            if self.coref_module_activate:
                ambiguity_list = self.find_reference_ambiguity(utterance_)
                if self.verbose:
                    print("Pronouns are: ", ambiguity_list, "\n")
                
                ref_new_utterance = None
                for ambiguity_info in ambiguity_list:
                    if self.add_oriq_in_context:
                        marked_query = self.mark_ambiguity_word(utterance_, ambiguity_info)
                        context = context + ' ' + marked_query
                    answers = self.find_qa_results(ambiguity_type='pronoun', ambiguity_info=ambiguity_info, raw_utterance=utterance_, context=context, question_answer_pip=self.ref_pipl)
                    if answers == []:
                        pass
                    else:
                        answer = self.anwsers_selection(utterance_, utterance_, answers)
                        if answer == '':
                            pass
                        else:
                            ref_new_utterance = self.rewriting(answer, 'pronoun', ambiguity_info, utterance_)                        
                        #utterance_ = new_utterance
            else:
                ref_new_utterance = None
            
            
            
            if self.omission_module_activate == False:
                des_verb_new_utterance = None
                des_noun_new_utterance = None
            else:
                ## 2nd find des_noun ambiguity
                des_noun_new_utterance = None
                ambiguity_list, bm25scores = self.find_des_noun_ambiguity(utterance_)
                if self.verbose:
                    print("\nThe ambiguity nouns are: ", ambiguity_list, 
                        "\nwith bm25 scores: ", bm25scores, '\n')
                if len(ambiguity_list) != 0:
                    count = 0
                    for ambiguity_i, ambiguity_info in enumerate(ambiguity_list):
                    #ambiguity_info = ambiguity_list[0]

                        if ambiguity_info != '':
                            if self.add_oriq_in_context:
                                marked_query = self.mark_ambiguity_word(utterance_, ambiguity_info)
                                context = context + ' ' + marked_query
                            answers = self.find_qa_results(ambiguity_type='des_noun', ambiguity_info=ambiguity_info, raw_utterance=utterance_, context=context, question_answer_pip=self.des_pipl)         
                            # not an existing information in the utterance (step 1 did not edit anything)
                            if ref_new_utterance == None:
                                if answers == []:
                                    pass
                                else:
                                    answer = self.anwsers_selection(utterance_, utterance_, answers)
                                    if answer == '':
                                        pass
                                    else:                                    
                                        des_noun_new_utterance = self.rewriting(answer, 'des_noun', ambiguity_info, utterance_)
                            else:
                                if answers == []:
                                    pass
                                else:
                                    answer = self.anwsers_selection(utterance_, ref_new_utterance, answers)
                                    if answer == '':
                                        pass
                                    else:                                    
                                        des_noun_new_utterance = self.rewriting(answer, 'des_noun', ambiguity_info, ref_new_utterance)                        
                                
                        
                        count += 1
                        if count >= self.max_des_noun_ambiguity_loop:
                            break
                        
                
                ## 3rd find des_verb ambiguity
                des_verb_new_utterance = None
                ambiguity_list, bm25scores = self.find_des_verb_ambiguity(utterance_)
                if self.verbose:
                    print("\nThe ambiguity verbs are: ", ambiguity_list, 
                        "\nwith bm25 scores: ", bm25scores, '\n')
                if (len(ambiguity_list) != 0):
                    #ambiguity_info = ambiguity_list[0]
                    count = 0
                    for ambiguity_i, ambiguity_info in enumerate(ambiguity_list):
                        if ambiguity_info != '':
                            if self.add_oriq_in_context:
                                marked_query = self.mark_ambiguity_word(utterance_, ambiguity_info)
                                context = context + ' ' + marked_query
                            answers = self.find_qa_results(ambiguity_type='des_verb', ambiguity_info=ambiguity_info, raw_utterance=utterance_, context=context, question_answer_pip=self.des_pipl)
                            
                            # step 2 edited something
                            if des_noun_new_utterance != None:
                                
                                if answers == []:
                                    pass
                                else:
                                    answer = self.anwsers_selection(utterance_,des_noun_new_utterance, answers)
                                    if answer == '':
                                        pass
                                    else:                                    
                                        des_verb_new_utterance = self.rewriting(answer, 'des_verb', ambiguity_info, des_noun_new_utterance)
                            
                            # step 1 edited something, but step 2 did not edit anything
                            elif ref_new_utterance != None:
                                if answers == []:
                                    pass
                                else:
                                    answer = self.anwsers_selection(utterance_,ref_new_utterance, answers)
                                    if answer == '':
                                        pass
                                    else:                                    
                                        des_verb_new_utterance = self.rewriting(answer, 'des_verb', ambiguity_info, ref_new_utterance)
                            
                            # step 1 and step 2 either did not edit anything
                            else:
                                if answers == []:
                                    pass
                                else:
                                    answer = self.anwsers_selection(utterance_,utterance_, answers)
                                    if answer == '':
                                        pass
                                    else:                                    
                                        des_verb_new_utterance = self.rewriting(answer, 'des_verb', ambiguity_info, utterance_)
                            
                                #utterance_ = new_utterance
                        
                        count += 1
                        if count >= self.max_des_verb_ambiguity_loop:
                            break
        
        
        
        
        
        
            # priority usage= des_verb > des_noun > ref
            if des_verb_new_utterance != None:
                ultimate = des_verb_new_utterance
            elif des_noun_new_utterance != None:
                ultimate = des_noun_new_utterance
            elif ref_new_utterance != None:
                ultimate = ref_new_utterance
            else:
                ultimate = query
            if self.verbose:
                print("\nThe ultimated utterance is: ", ultimate, '\n')
        
            return ultimate
    
    
    def clean_raw_utterance(self, utterance):
        '''
        remove punctuations and extra spaces
        '''
        utterance_ = utterance
        utterance_ = ' '+utterance_
        #punct = string.punctuation.replace("'", "") # remove ' from punct
        punct = ".,‚?!'"
        for p in punct:
            utterance_ = utterance_.replace(p, ' '+p)
        
        return utterance_
          
    def anwsers_selection(self, raw_utterance, utterance, answers):
        '''
        Intro:
            select the answer from the answers list
            
        Input:
            - utterance: str
            - answers: list of str
            
        Return:
            - answer: str
        '''
        #if self.select_top_answer:
        info_tobe_added = ''
        for answer in answers:
            answer = answer.translate(str.maketrans('', '', string.punctuation)) # remove special characters
            if (answer in utterance) or (answer in raw_utterance):
                pass
            else:
                info_tobe_added = answer
                break
        return info_tobe_added
                
            
            
            
            
        
        

    def find_qa_results(self, ambiguity_type:str, ambiguity_info:str, raw_utterance:str, context:str, question_answer_pip)->list:
        '''
        Intro:
            using the question_answer_pip to find the answer
        
        Return:
            - WARNING: not a list but a str(the shortest answer in the top 5 max )
            - EDIT: return a list of answers
        
        
        Question thinking:
            - must add raw utterance to the question, instead of context
                - QA is trained never select content in the question
        '''
        
        
        
        #print(f"Resolving {ambiguity_type} ambiguity...towards {ambiguity_info}")
        if ambiguity_type in ['des_noun', 'des_verb']:
            if ambiguity_type == 'des_noun':
                if self.add_oriq_in_question:
                    ask_q = f'{self.mark}{ambiguity_info}{self.mark} of what? in "{raw_utterance}".'
                else:
                    ask_q = f'{self.mark}{ambiguity_info}{self.mark} of what?'
            elif ambiguity_type == 'des_verb':
                if self.add_oriq_in_question:
                    ask_q = f'{self.mark}{ambiguity_info}{self.mark} to what? in "{raw_utterance}".'
                else:
                    ask_q = f'{self.mark}{ambiguity_info}{self.mark} to what?'
            else:
                pass
            results = question_answer_pip(question=ask_q,  context=context, topk=self.search_in_qa_topk)
        elif ambiguity_type == 'pronoun':
            if self.add_oriq_in_question:
                ask_q = f'What is {self.mark}{ambiguity_info}{self.mark} refer to? in "{raw_utterance}".'
            else:
                ask_q = f'What is {self.mark}{ambiguity_info}{self.mark} refer to?'
            results = question_answer_pip(question=ask_q,  context=context, topk=self.search_in_qa_topk)
        
        if self.verbose:
            print("asking question: ", ask_q)

        ### collect answers
        answers = []
        for result in results:
            #print(result)
            ##### Rule1: the length should not succceed the max_length
            if len(result['answer'].split(' ')) > self.answer_max_length:
                continue
            ##### Rule2: the info should not contains any punctuations
            if not self.filter_punct(result['answer']):
                continue
            answers.append(result['answer'])
        
        
        if self.verbose:
            print(f"\nBy probability, The answers of {ambiguity_type} are: ", answers, "\n")
        
        
        ###### sort the answers by length (default by probability)    
        if self.sort_by_length:
            answers.sort(key=lambda x: len(x.split(' '))) # sort by length
        
        
        
        if len(answers) == 0:
            return []
        else:
            return answers

    def filter_punct(self, utterance:str) -> bool:
        focal_puncts = ".,?!:"
        good = 0
        for punct in focal_puncts:
            if punct not in utterance:
                good += 1
        
        return good == len(focal_puncts)
        

    def rewriting(self, answer: str, ambiguity_type:str, ambiguity_info: str, raw_utterance: str):
        
        '''
        Intro:
            rewrite the utterance based on the ambiguity type and the answers
        
        WARINING:
            - ambiguity_info = str, not a list of str
        
        '''
        #### add some white space to the raw_utterance to make sure the ambiguity_info is not a substring of other words

        if answer not in raw_utterance:
            if ambiguity_type == 'pronoun':
                # replace the pronoun with the answer
                if self.verbose:
                    print(f"-> replacing '{ambiguity_info}' with '{answer}'")
                
                # if ambiguity_info in the middle of the sentence
                new_utterance = raw_utterance.replace(' '+ambiguity_info+' ', ' '+answer+' ', 1) # make sure it is a word (with space)
                
                
            elif ambiguity_type in ['des_noun', 'des_verb']:
                # replace the noun/verb with the answer
                if ambiguity_type == 'des_noun':
                    additonal_info = 'of'
                elif ambiguity_type == 'des_verb':
                    additonal_info = 'to'
                if self.verbose:
                    print("-> adding additional info: '", answer, "', after: '", ambiguity_info, "'")
                new_utterance = raw_utterance.replace(' '+ambiguity_info+' ', ' '+ambiguity_info + f' {additonal_info} ' + answer+' ', 1)
            return new_utterance
        else:
            return raw_utterance
        
    def find_reference_ambiguity(self, utterance:str)-> List[str]:
        used_pronouns = self.find_pronoun(self.overallpronouns, utterance) # return a list
        return used_pronouns

    def find_des_noun_ambiguity(self, utterance:str)-> List[str]:
        '''
        Find nouns in the utterance that is :
            - important (high bm25 scores)
            - does not have descriptive information after it
        
        '''
        imp_nouns = []
        imp_nouns_bm25_scores = []
        for i in range(0, self.max_topk):
            try: # some times the length of bm25 scores list < max_topk, will arise "list index out of range" IndexError
                focal_word, score = self.find_important_noun(pipl=self.pos_pipl, possible_important_nouns=self.possible_important_nouns, searcher=self.searcher, sentence=utterance, overallpronouns=self.overallpronouns, topk=i)
            except:
                break
            ##### check if 1. the noun is already in the list 2. the noun is not empty 3. the noun bm25 score is not very small
            if (focal_word not in imp_nouns) and (focal_word != '') and (score > self.minimum_bm25_score):
                imp_nouns.append(focal_word)
                imp_nouns_bm25_scores.append(score)
        
        return imp_nouns, imp_nouns_bm25_scores

    def find_des_verb_ambiguity(self, utterance:str)-> List[str]:
        '''
        Find verbs in the utterance that is :
            - important (high bm25 scores)
            - does not have descriptive information after it
        
        
        '''
        # if no nouns, find verbs
        imp_verbs = []
        imp_verbs_bm25_scores = []
        for i in range(0, self.max_topk):
            try:
                focal_word, score = self.find_important_verb(pipl=self.pos_pipl, searcher=self.searcher, sentence=utterance, topk=i)
            except:
                break
            if (focal_word not in imp_verbs) and (focal_word != '') and (score > self.minimum_bm25_score):
                imp_verbs.append(focal_word)
                imp_verbs_bm25_scores.append(score)
        
        return imp_verbs, imp_verbs_bm25_scores
        
        
    def find_ambiguity(self, question:str):
        '''
        Intro:
            find ambiguity in a question
            - Priority: pronoun > des_noun > des_verb
            
            
        Return:
            - ambiguity_type: str
            - ambiguity_list: list of str
        
        '''
        ambiguity_type = ''
        # first, try to find pronouns
        used_pronouns = self.find_pronoun(self.overallpronouns, question) # return a list
        
        if len(used_pronouns) != 0:
            ambiguity_type = 'pronoun'
            return ambiguity_type, used_pronouns # return a list
        
        else:
            # if no pronouns, find nouns
            imp_nouns = []
            for i in range(1, self.max_topk+1):
                try: # some times the length of bm25 scores list < max_topk, will arise "list index out of range" IndexError
                    focal_word, score = self.find_important_noun(pipl=self.pos_pipl, possible_important_nouns=self.possible_important_nouns, searcher=self.searcher, sentence=question, overallpronouns=self.overallpronouns, topk=i)
                except:
                    break
                if focal_word not in imp_nouns:
                    imp_nouns.append(focal_word)
                
            
            if len(imp_nouns) != 0:
                ambiguity_type = 'des_noun'
                return ambiguity_type, imp_nouns # return a list
            
            else:
                # if no nouns, find verbs
                imp_verbs = []
                for i in range(1, self.max_topk+1):
                    try:
                        focal_word, score = self.find_important_verb(pipl=self.pos_pipl, searcher=self.searcher, sentence=question, topk=i)
                    except:
                        break
                    if focal_word not in imp_verbs:
                        imp_verbs.append(focal_word)
                    
                    
                    
                if len(imp_verbs) != 0:
                    ambiguity_type = 'des_verb'
                    return ambiguity_type, imp_verbs
                else:
                    return ambiguity_type, []
        
    
    
    def build_pipeline(self):
        self.pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=self.device)



    def find_pronoun(self, all_pronouns: list, sentence:str) -> list:
        '''
        Intro:
            - find all pronouns in the sentence
        
        '''
        used_pronouns = []
        for pronoun in all_pronouns:
            if pronoun in sentence.split(' '):
                used_pronouns.append(pronoun)
        return used_pronouns

    def find_important_noun(self, pipl:TokenClassificationPipeline, possible_important_nouns: list, searcher:SimpleSearcher, sentence:str, overallpronouns:list, topk:int) -> str:
        '''
        Intro:
            - return the topk_th important noun(with top BM25 scores) in the sentence
                - Only 1 noun is returned!!!!!
            - cannot return useless Pronouns 
                - e.g. what, it, they, etc.
                - useful pronouns: Apple, Google, Goeing, John Hobber, etc.
            
        Input:
            - top k start with 0
        '''
        
        abandone_results = ['what', 'it'] + overallpronouns
        
        # get verbs
        outputs = pipl(sentence)
        verb_outputs = []
        which_outputs = []
        for i, output in enumerate(outputs):
            if output['entity'] in possible_important_nouns:
                verb_outputs.append(output)
                which_outputs.append(i)
        
                
                
        # get bm25 score
        bm25_scores = []
        for n_output in verb_outputs:
            focal_noun = n_output['word']
            try:
                bm25_s = searcher.search(focal_noun, k=1)[0].score
            except:
                bm25_s = 0
            bm25_scores.append(bm25_s)
        
        if len(bm25_scores) == 0:
            return ''
        else:
            sorted_bm25_scores = sorted(bm25_scores, reverse=True)
            idx = [bm25_scores.index(x) for x in sorted(bm25_scores, reverse=True)][topk]
            score = sorted_bm25_scores[topk]
            candidate = verb_outputs[idx]['word']
            
            # avoid IndexError (for focal word is the last one in the qn)
            try:
                candidate_next = verb_outputs[idx+1]['word']
            except:
                candidate_next = ''
            
            # candidate is a subfix
            if candidate.startswith('##'):
                for i in range(idx-1, -1, -1):
                    if verb_outputs[i]['word'].startswith('##'):
                        continue
                    else:
                        # if the candidate is also a prefix
                        if candidate_next.startswith('##'):
                            for j in range(idx+1, len(verb_outputs)):
                                if verb_outputs[j]['word'].startswith('##'):
                                    continue
                                else:
                                    imp_noun = sentence[verb_outputs[i]['start']:verb_outputs[j-1]['end']]
                                    break
                        else:
                            imp_noun = sentence[verb_outputs[i]['start']:verb_outputs[idx]['end']]
                        break
            
            # candidate is a prefix(cannot be a subfix)
            elif candidate_next.startswith('##'): 
                for i in range(idx+1, len(verb_outputs)):
                    if verb_outputs[i]['word'].startswith('##'):
                        continue
                    else:
                        imp_noun = sentence[verb_outputs[idx]['start']:verb_outputs[i-1]['end']]
                        break
            else:
                imp_noun = candidate
            
            if imp_noun in abandone_results:
                imp_noun = ''
            return imp_noun, score
            
                
    def find_important_verb(self, pipl:TokenClassificationPipeline, searcher: SimpleSearcher, sentence: str, topk:int)->str:
        '''
        Intro:
            - return the topk_th important verb in the sentence
                - Only 1 verb is returned
            - top k start with 0
        '''
        # get verbs
        outputs = pipl(sentence)
        verb_outputs = []
        which_outputs = []
        for i, output in enumerate(outputs):
            if output['entity'] in ['VERB']:
                verb_outputs.append(output)
                which_outputs.append(i)
        
                
                
        # get bm25 score
        bm25_scores = []
        for n_output in verb_outputs:
            focal_noun = n_output['word']
            try:
                bm25_s = searcher.search(focal_noun, k=1)[0].score
            except:
                bm25_s = 0
            bm25_scores.append(bm25_s)
        
        if len(bm25_scores) == 0:
            return ''
        else:
            sorted_bm25_scores = sorted(bm25_scores, reverse=True)
            idx = [bm25_scores.index(x) for x in sorted(bm25_scores, reverse=True)][topk]
            score = sorted_bm25_scores[topk]
            candidate = verb_outputs[idx]['word']
            
            # avoid IndexError (for focal word is the last one in the qn)
            try:
                candidate_next = verb_outputs[idx+1]['word']
            except:
                candidate_next = ''
            
            # candidate is a subfix
            if candidate.startswith('##'):
                for i in range(idx-1, -1, -1):
                    if verb_outputs[i]['word'].startswith('##'):
                        continue
                    else:
                        # if the candidate is also a prefix
                        if candidate_next.startswith('##'):
                            for j in range(idx+1, len(verb_outputs)):
                                if verb_outputs[j]['word'].startswith('##'):
                                    continue
                                else:
                                    imp_verb = sentence[verb_outputs[i]['start']:verb_outputs[j-1]['end']]
                                    break
                        else:
                            imp_verb = sentence[verb_outputs[i]['start']:verb_outputs[idx]['end']]
                        break
            
            # candidate is a prefix
            elif candidate_next.startswith('##'): 
                for i in range(idx+1, len(verb_outputs)):
                    if verb_outputs[i]['word'].startswith('##'):
                        continue
                    else:
                        imp_verb = sentence[verb_outputs[idx]['start']:verb_outputs[i-1]['end']]
                        break
            else:
                imp_verb = candidate
            
            return imp_verb, score