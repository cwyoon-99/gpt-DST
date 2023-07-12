import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
import copy
import time
from collections import defaultdict
from tqdm import tqdm
from utils.helper import SpeedLimitTimer, PreviousStateRecorder
from utils.typo_fix import typo_fix
from config import CONFIG

from api_request.gpt35_turbo_completion import gpt35_turbo_completion, gpt35_turbo_completion_with_usage
from utils.our_parse import sv_dict_to_string, our_pred_parse, our_pred_parse_with_bracket, pred_parse_with_bracket_matching, active_domain_parse
from prompt.our_prompting import conversion, get_our_prompt, custom_prompt, get_prompt_with_bracket, \
    cluster_print, select_active_domain, active_domain_ontology
from retriever.code.embed_based_retriever import EmbeddingRetriever
from evaluate.evaluate_metrics import evaluate
from evaluate.evaluate_FGA import FGA

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)", required=True)  # e.g. "./data/mw21_10p_train_v3.json"
parser.add_argument('--retriever_dir', type=str, required=True, help="sentence transformer saved path")  # "./retriever/expts/mw21_10p_v3_0304_400_20"
parser.add_argument('--output_file_name', type=str, default="debug", help="filename to save running log and configs")
parser.add_argument('--output_dir', type=str, default="./expts/debug", help="dir to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.1", choices=['2.1', '2.4'], help="version of MultiWOZ")  
parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
parser.add_argument('--save_interval', type=int, default=5, help="interval to save running_log.json")
parser.add_argument('--test_size', type=int, default=10, help="size of the test set")
parser.add_argument('--bracket', action="store_true", help="whether brackets are used in each domain-slot")
parser.add_argument('--slot_classify', action="store_true", help="whether slots are predicted through index number")

# cluster
parser.add_argument('--cluster', action="store_true", help="whether examples are clustered or not")
parser.add_argument('--num_cand', type=int, default=10)
parser.add_argument('--num_select', type=int, default=5)
parser.add_argument('--num_max', type=int)
parser.add_argument('--num_min', type=int)
parser.add_argument('--domain_select', action="store_true", help="whether to select domain in ontology")

args = parser.parse_args()

# current time
cur_time = time.strftime('%y%m%d_%H%M-')

# create the output folder
args.output_dir = 'expts/' + cur_time + args.output_file_name + '_0to' + str(args.test_size)
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

if args.num_cand: N_CAND=args.num_cand
if args.num_select: N_SELECT=args.num_select

# read the selection pool
with open(args.train_fn) as f:
    train_set = json.load(f)

# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
    if args.test_fn == "":
        test_set_path = "./data/mw21_100p_test.json"
else:
    ontology_path = CONFIG["ontology_24"]
    if args.test_fn == "":
        test_set_path = "./data/mw24_100p_test.json"

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

with open(ontology_path) as f:
    ontology = json.load(f)
with open(test_set_path) as f:
    test_set = json.load(f)

# load the retriever
retriever = EmbeddingRetriever(datasets=[train_set],
                               model_path=args.retriever_dir,
                               search_index_filename=os.path.join(args.retriever_dir, "train_index.npy"), 
                               sampling_method="pre_assigned")

def run(test_set, turn=-1, use_gold=False):
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    timer = SpeedLimitTimer(second_per_step=3.1)  # openai limitation 20 queries/min

    result_dict = defaultdict(list)  # use to record the accuracy

    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError("can only evaluate particular turn when using gold context")
        selected_set = [d for d in test_set if len(d['dialog']['usr']) == turn + 1]
    
    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    # usage measure
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_n_example = 0

    # specify ontology_prompt, prompt_function
    mode = "default"
    if args.bracket:
        ontology_prompt = custom_prompt
        get_prompt = get_prompt_with_bracket
        our_parse = pred_parse_with_bracket_matching

    for data_idx, data_item in enumerate(tqdm(selected_set)):
        n_total += 1

        completion = ""
        if use_gold:
            prompt_text = get_prompt(
                data_item, examples=retriever.item_to_nearest_examples(data_item, k=NUM_EXAMPLE))
        else:
            predicted_context = prediction_recorder.state_retrieval(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context

        if args.cluster:
            orig_examples = retriever.item_to_nearest_examples(
                modified_item, k=N_CAND)

            examples, cluster_info = retriever.dynamic_cluster_examples(
                modified_item, k=N_CAND, max_ex = args.num_max, min_ex = args.num_min
            )

            data_item['cluster_info'] = cluster_print(orig_examples, cluster_info)

        else:
            examples = retriever.item_to_nearest_examples(
                modified_item, k=N_SELECT)


        data_item['n_example'] = len(examples)
        total_n_example += data_item['n_example']

        prompt_text = get_prompt(
                data_item, examples=examples, given_context=predicted_context)

        if args.domain_select:
            # select active domain
            select_domain_prompt = select_active_domain(data_item)
            
            complete_flag = False
            while not complete_flag:
                try:
                    active_domain_completion, domain_usage = gpt35_turbo_completion_with_usage(select_domain_prompt)
                except Exception as e:
                    print(e)
                    # throughput too high
                    timer.sleep(10)
                else:
                    complete_flag = True
                # limit query speed
                timer.step()

            data_item['domain_completion'] = active_domain_completion
            prompt_tokens += domain_usage["prompt_tokens"]
            completion_tokens += domain_usage["completion_tokens"]
            total_tokens += domain_usage["total_tokens"]

            pred_active_domain = active_domain_parse(active_domain_completion)
            
            # add extra domains from retreived examples
            for example in examples:
                for s,v in example['turn_slot_values'].items():
                    pred_active_domain.add(s.split('-', 1)[0])
            
            # save
            label_domain = set([s.split('-', 1)[0] for s,v in data_item['turn_slot_values'].items()])
            if label_domain.issubset(pred_active_domain):
                data_item['pred_domain_status'] = "correct"
                print("pred domain correct\n")
            else:
                data_item['pred_domain_status'] = "wrong"
                print("pred domain wrong\n")

            data_item['predict_domain'] = list(pred_active_domain)
            data_item['label_domain'] = list(label_domain)
            print(f"predict_domain: {data_item['predict_domain']}")
            print(f"  label domain:  {data_item['label_domain']}")
            
            
            domain_selected_ontology = active_domain_ontology(ontology_prompt, pred_active_domain)
            prompt_text = prompt_text.replace(conversion(ontology_prompt), conversion(domain_selected_ontology))

        print(prompt_text.replace(conversion(ontology_prompt), ""))

        # record the prompt
        data_item['prompt'] = prompt_text

        # gpt35 completion
        complete_flag = False
        parse_error_count = 0
        while not complete_flag:
            try:
                completion, usage = gpt35_turbo_completion_with_usage(prompt_text)
                completion = conversion(completion, reverse=True)
            except Exception as e:
                if e.user_message.startswith("This model's maximum context length"):
                    print("prompt overlength")
                    examples = examples[1:]
                    prompt_text = get_prompt(
                        data_item, examples=examples, given_context=predicted_context)
                else:
                    # throughput too high
                    timer.sleep(10)
            else:
                try:
                    # check if CODEX is crazy 
                    temp_parse = our_parse(completion)
                except:
                    print("parse error")
                    print("generate completion again...")
                    parse_error_count += 1
                    if parse_error_count >= 3:
                        print("exceed parse error limit... exit")
                        complete_flag = True
                else:
                    complete_flag = True
            # limit query speed
            timer.step()

        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = our_parse(completion) # a dictionary
        except:
            print("the output is not a valid result")
            data_item['not_valid'] = 1

        predicted_slot_values = typo_fix(predicted_slot_values, ontology=ontology, version=args.mwz_ver)

        context_slot_values = data_item['last_slot_values']  # a dictionary

        # merge context and prediction
        if use_gold:
            all_slot_values = context_slot_values.copy()
        else:
            all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()

        for s, v in predicted_slot_values.items():
            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}
        
        prediction_recorder.add_state(data_item, all_slot_values)

        # record the predictions
        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = ontology_path
        data_item['completion'] = completion
        data_item['predicted_slot_values'] = predicted_slot_values

        # print the result
        print(completion)
        print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
        print(f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        print(f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(all_slot_values,data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
            print("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']].append(0)
            print("\n=====================wrong!=======================")

        # save result
        data_item['JGA'] = n_correct / n_total
        data_item['SA'] = total_acc / n_total
        data_item['Joint_F1'] = total_f1 / n_total
        data_item['pred_status'] = 'correct' if this_jga else 'wrong'

        all_result.append(data_item)

        # Log Checkpoint
        if data_idx % args.save_interval == 0:
            with open(os.path.join(args.output_dir,f'running_log.json'),'w') as f:
                json.dump(all_result, f, indent=4)

        print("\n\n\n####################################################################################################################\n\n\n")

    print(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    print(f"Slot Acc {total_acc/n_total}")
    print(f"Joint F1 {total_f1/n_total}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")

    # save score in score.txt
    with open(os.path.join(args.output_dir, "score.txt"), 'w') as f:
        f.write(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}\n")
        f.write(f"Slot Acc {total_acc/n_total}\n")
        f.write(f"Joint F1 {total_f1/n_total}\n")
         # calculate the accuracy of each turn
        for k, v in result_dict.items():
            f.write(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}\n")

        f.write(f"\nprompt_tokens: {prompt_tokens}\n")
        f.write(f"completion_tokens: {completion_tokens}\n")
        f.write(f"total_tokens: {total_tokens}\n\n")

        f.write(f"total_n_example: {total_n_example}\n")
        f.write(f"average_n_example: {total_n_example/(data_idx + 1)}\n\n")

    return all_result


if __name__ == "__main__":

    # api 사용량 위해 개수 제한
    if args.test_size != -1:
        test_set = test_set[:args.test_size]
    all_results = run(test_set)

    with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # save FGA in score.txt
    with open(os.path.join(args.output_dir, "score.txt"), 'a') as f:
        fga_result = FGA(os.path.join(args.output_dir, "running_log.json"))
        f.write("\nFGA Result\n")
        f.write("\n".join(fga_result))
    
    print(f"End time: {time.strftime('%y%m%d_%H%M')}")