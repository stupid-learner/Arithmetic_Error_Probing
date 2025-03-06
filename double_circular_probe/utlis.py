import random
import csv

def random_select_tuples(tuple_list, n, seed = 42):
    #randomly select n prompts
    if n > len(tuple_list):
        raise ValueError("n cannot be greater than the length of the tuple list")
    if seed is not None:
        random.seed(seed)
    return random.sample(tuple_list, n)


def load_model_result_dic(lower_bound = 0, sum_upper_bound = 1000):
    folder = "arithmetic_data"
    csv_files = []
    for i in range(1,10):
        csv_files.append("data_{}00_to_{}00".format(i,i+1))
    result = []

    for csv_file in csv_files:
        with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            i = 0
            for row in reader:
                result.append(tuple(row))
                i+=1

    filtered_result = [i for i in result if (not "(" in i[2] and not "+" in i[2] and 3 <= len(i[2]) <= 4)]

    result_dic = {}

    for i in filtered_result:
        if int(i[0])+int(i[1]) < sum_upper_bound and int(i[2]) < sum_upper_bound:
            result_dic[(int(i[0]),int(i[1]))] = int(i[2])

    return result_dic

def get_balanced_data(prompts, num_per_digit = 100):
    balanced_prompts = []
    prompt_by_starting_digit = [[] for _ in range(10)]
    for i in prompts:
        prompt_by_starting_digit[(i[0]+i[1])//100].append(i)
    for i in range(10):
        if len(prompt_by_starting_digit[i]) >= num_per_digit:
            balanced_prompts += random_select_tuples(prompt_by_starting_digit[i], num_per_digit)

    return balanced_prompts


