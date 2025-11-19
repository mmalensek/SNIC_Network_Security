import csv
import random
import pandas as pd
from ollama import chat

# preparing the prompt
def create_prompt(record, type):
    # for now you have to manually set the prompt
    if(type == "TEST"):
        prompt = f"""Network record details: {record}. Is this traffic malicious? Answer "BENIGN" or "DDoS"."""
    if(type == "START"):
        prompt = """You are to analyze traffic flow data and determine if the traffic is benign or it represents part of a DDoS attack. Each flow is described with multiple features such as source and destination IPs and ports, protocol, timestamps, packet counts and lengths, flow duration, packet inter-arrival times, flags, bytes per second, and more. Some flows represent benign network activity and some represent DDoS attacks. 
                    Here are the columns: 
                    Flow ID	Source IP	Source Port	Destination IP	Destination Port	Protocol	Timestamp	Flow Duration	Total Fwd Packets	Total Backward Packets	Total Length of Fwd Packets	Total Length of Bwd Packets	Fwd Packet Length Max	Fwd Packet Length Min	Fwd Packet Length Mean	Fwd Packet Length Std	Bwd Packet Length Max	Bwd Packet Length Min	Bwd Packet Length Mean	Bwd Packet Length Std	Flow Bytes/s	Flow Packets/s	Flow IAT Mean	Flow IAT Std	Flow IAT Max	Flow IAT Min	Fwd IAT Total	Fwd IAT Mean	Fwd IAT Std	Fwd IAT Max	Fwd IAT Min	Bwd IAT Total	Bwd IAT Mean	Bwd IAT Std	Bwd IAT Max	Bwd IAT Min	Fwd PSH Flags	Bwd PSH Flags	Fwd URG Flags	Bwd URG Flags	Fwd Header Length	Bwd Header Length	Fwd Packets/s	Bwd Packets/s	Min Packet Length	Max Packet Length	Packet Length Mean	Packet Length Std	Packet Length Variance	FIN Flag Count	SYN Flag Count	RST Flag Count	PSH Flag Count	ACK Flag Count	URG Flag Count	CWE Flag Count	ECE Flag Count	Down/Up Ratio	Average Packet Size	Avg Fwd Segment Size	Avg Bwd Segment Size	Fwd Header Length	Fwd Avg Bytes/Bulk	Fwd Avg Packets/Bulk	Fwd Avg Bulk Rate	Bwd Avg Bytes/Bulk	Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate	Subflow Fwd Packets	Subflow Fwd Bytes	Subflow Bwd Packets	Subflow Bwd Bytes	Init_Win_bytes_forward	Init_Win_bytes_backward	act_data_pkt_fwd	min_seg_size_forward	Active Mean	Active Std	Active Max	Active Min	Idle Mean	Idle Std	Idle Max	Idle Min	Label
                    Next I am going to include non-labeled traffic and you are to determine what kind of traffic it is. Answer ONLY with either “BENIGN” or “DDoS”."""
    return prompt

# evaluating the results
def evaluate_results(numTest, numCorrect):
    return numCorrect / numTest

# get the number of rows in a dataset
def getDataSetHeight(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rowCount = sum(1 for row in reader)
    return rowCount

# get the number of columns in a dataset
def getDataSetWidth(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        first_row = next(reader)  # read the first row (header)
        return len(first_row)

def run_tests(dataset, labelIndex, numberTests, model):
    random.seed(42)
    datasetHeight = len(dataset)
    sample_indexes = random.sample(range(datasetHeight), numberTests)
    numCorrect = 0

    # give the first prompt (start)
    prompt = create_prompt("", "START")
    messages = [{"role": "user", "content": prompt}]
    response = chat(model=model, messages=messages)

    for idx in sample_indexes:
        row = dataset.iloc[idx]
        record = row.drop(labelIndex).tolist()
        prompt = create_prompt(record, "TEST")

        messages = [{"role": "user", "content": prompt}]
        response = chat(model=model, messages=messages)

        ai_answer = response["message"]["content"].strip().lower()
        true_label = str(row[labelIndex]).strip().lower()

        if ai_answer == true_label:
            numCorrect += 1

        print(f"Test #{idx}: AI answer = {ai_answer}, True label = {true_label}")

    return numCorrect

def main():

    # TEMP FILEPATH AND DELIMITER VALUES
    # filepath = "../../../dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    filepath = "../../dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    delimiter = ","
    dataset = pd.read_csv(filepath, delimiter=delimiter)

    print("--------------------------------------")
    datasetHeight = getDataSetHeight(filepath)
    datasetWidth = getDataSetWidth(filepath)
    labelIndex = datasetWidth - 1
    label_values = dataset.iloc[:, labelIndex].unique()
    print("Unique label values:", label_values)
    print("Number of rows in the dataset:", datasetHeight)
    print("--------------------------------------")

    numberTests = int(input("\nSet the number of tests: "))
    model = input("Select the wanted model: (deepseek-r1:32b, gpt-oss:20b, gemma3:1b, ...): ")
    numCorrect = run_tests(dataset, labelIndex, numberTests, model)

    accuracy = evaluate_results(numberTests, numCorrect)
    print("\n--------------------------------------")
    print(f"Accuracy over {numberTests} tests: {accuracy:.2%}")
    print("--------------------------------------")


if __name__ == "__main__":
    main()
