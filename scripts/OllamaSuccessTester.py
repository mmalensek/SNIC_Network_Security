import csv
import random
import pandas as pd
from ollama import chat

# TO-DO
# dodat moznost za razlicne datasete - trenutno omejen na ddos dataset
# dodat uravnotezene flowe, kjer je 50% benign in 50% ostalih drugacnih

# preparing the prompt
def create_prompt(record, type):
    # for now you have to manually set the prompt
    if(type == "TEST DDOS"):
        prompt = f"""Network record details: {record}. Is this traffic malicious? Answer "BENIGN" or "DDoS"."""
    if(type == "START DDOS ZERO-SHOT"):
        prompt = """You are to analyze traffic flow data and determine if the traffic is benign or it represents part of a DDoS attack. Each flow is described with multiple features such as source and destination IPs and ports, protocol, timestamps, packet counts and lengths, flow duration, packet inter-arrival times, flags, bytes per second, and more. Some flows represent benign network activity and some represent DDoS attacks. 
                    Here are the columns: 
                    Flow ID	Source IP	Source Port	Destination IP	Destination Port	Protocol	Timestamp	Flow Duration	Total Fwd Packets	Total Backward Packets	Total Length of Fwd Packets	Total Length of Bwd Packets	Fwd Packet Length Max	Fwd Packet Length Min	Fwd Packet Length Mean	Fwd Packet Length Std	Bwd Packet Length Max	Bwd Packet Length Min	Bwd Packet Length Mean	Bwd Packet Length Std	Flow Bytes/s	Flow Packets/s	Flow IAT Mean	Flow IAT Std	Flow IAT Max	Flow IAT Min	Fwd IAT Total	Fwd IAT Mean	Fwd IAT Std	Fwd IAT Max	Fwd IAT Min	Bwd IAT Total	Bwd IAT Mean	Bwd IAT Std	Bwd IAT Max	Bwd IAT Min	Fwd PSH Flags	Bwd PSH Flags	Fwd URG Flags	Bwd URG Flags	Fwd Header Length	Bwd Header Length	Fwd Packets/s	Bwd Packets/s	Min Packet Length	Max Packet Length	Packet Length Mean	Packet Length Std	Packet Length Variance	FIN Flag Count	SYN Flag Count	RST Flag Count	PSH Flag Count	ACK Flag Count	URG Flag Count	CWE Flag Count	ECE Flag Count	Down/Up Ratio	Average Packet Size	Avg Fwd Segment Size	Avg Bwd Segment Size	Fwd Header Length	Fwd Avg Bytes/Bulk	Fwd Avg Packets/Bulk	Fwd Avg Bulk Rate	Bwd Avg Bytes/Bulk	Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate	Subflow Fwd Packets	Subflow Fwd Bytes	Subflow Bwd Packets	Subflow Bwd Bytes	Init_Win_bytes_forward	Init_Win_bytes_backward	act_data_pkt_fwd	min_seg_size_forward	Active Mean	Active Std	Active Max	Active Min	Idle Mean	Idle Std	Idle Max	Idle Min	Label
                    Next I am going to include non-labeled traffic and you are to determine what kind of traffic it is. Answer ONLY with either “BENIGN” or “DDoS”."""
    if(type == "START DDOS FEW-SHOT"):
        prompt = """You are to analyze traffic flow data and determine if the traffic is benign or it represents part of a DDoS attack. Here is network traffic flow data. Each flow is described with multiple features such as source and destination IPs and ports, protocol, timestamps, packet counts and lengths, flow duration, packet inter-arrival times, flags, bytes per second, and more. Some flows represent benign network activity and some represent DDoS attacks. 

Here are the columns: 

Flow ID	Source IP	Source Port	Destination IP	Destination Port	Protocol	Timestamp	Flow Duration	Total Fwd Packets	Total Backward Packets	Total Length of Fwd Packets	Total Length of Bwd Packets	Fwd Packet Length Max	Fwd Packet Length Min	Fwd Packet Length Mean	Fwd Packet Length Std	Bwd Packet Length Max	Bwd Packet Length Min	Bwd Packet Length Mean	Bwd Packet Length Std	Flow Bytes/s	Flow Packets/s	Flow IAT Mean	Flow IAT Std	Flow IAT Max	Flow IAT Min	Fwd IAT Total	Fwd IAT Mean	Fwd IAT Std	Fwd IAT Max	Fwd IAT Min	Bwd IAT Total	Bwd IAT Mean	Bwd IAT Std	Bwd IAT Max	Bwd IAT Min	Fwd PSH Flags	Bwd PSH Flags	Fwd URG Flags	Bwd URG Flags	Fwd Header Length	Bwd Header Length	Fwd Packets/s	Bwd Packets/s	Min Packet Length	Max Packet Length	Packet Length Mean	Packet Length Std	Packet Length Variance	FIN Flag Count	SYN Flag Count	RST Flag Count	PSH Flag Count	ACK Flag Count	URG Flag Count	CWE Flag Count	ECE Flag Count	Down/Up Ratio	Average Packet Size	Avg Fwd Segment Size	Avg Bwd Segment Size	Fwd Header Length	Fwd Avg Bytes/Bulk	Fwd Avg Packets/Bulk	Fwd Avg Bulk Rate	Bwd Avg Bytes/Bulk	Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate	Subflow Fwd Packets	Subflow Fwd Bytes	Subflow Bwd Packets	Subflow Bwd Bytes	Init_Win_bytes_forward	Init_Win_bytes_backward	act_data_pkt_fwd	min_seg_size_forward	Active Mean	Active Std	Active Max	Active Min	Idle Mean	Idle Std	Idle Max	Idle Min	Label

Below are examples of benign flows: 
192.168.10.16-104.17.241.25-46236-443-6	104.17.241.25	443	192.168.10.16	46236	6	7/7/2017 3:30	34	1	1	6	6	6	6	6	0	6	6	6	0	352941.1765	58823.52941	34	0	34	34	0	0	0	0	0	0	0	0	0	0	0	0	0	0	20	20	29411.76471	29411.76471	6	6	6	0	0	0	0	0	0	1	1	0	0	1	9	6	6	20	0	0	0	0	0	0	1	6	1	6	31	329	0	20	0	0	0	0	0	0	0	0	BENIGN
192.168.10.5-104.19.196.102-54863-443-6	104.19.196.102	443	192.168.10.5	54863	6	7/7/2017 3:30	3	2	0	12	0	6	6	6	0	0	0	0	0	4000000	666666.6667	3	0	3	3	3	3	0	3	3	0	0	0	0	0	0	0	0	0	40	0	666666.6667	0	6	6	6	0	0	0	0	0	0	1	0	0	0	0	9	6	0	40	0	0	0	0	0	0	2	12	0	0	32	-1	1	20	0	0	0	0	0	0	0	0	BENIGN
192.168.10.5-104.20.10.120-54871-443-6	104.20.10.120	443	192.168.10.5	54871	6	7/7/2017 3:30	1022	2	0	12	0	6	6	6	0	0	0	0	0	11741.68297	1956.947162	1022	0	1022	1022	1022	1022	0	1022	1022	0	0	0	0	0	0	0	0	0	40	0	1956.947162	0	6	6	6	0	0	0	0	0	0	1	0	0	0	0	9	6	0	40	0	0	0	0	0	0	2	12	0	0	32	-1	1	20	0	0	0	0	0	0	0	0	BENIGN

Below are examples of DDoS attack flows: 
172.16.0.1-192.168.10.50-57093-80-6	172.16.0.1	57093	192.168.10.50	80	6	7/7/2017 3:59	1488275	3	5	26	11607	20	0	8.666666667	10.26320288	5840	0	2321.4	3173.373883	7816.431775	5.375350658	212610.7143	562099.7266	1487333	3	685	342.5	422.1427484	641	44	1488212	372053	743520.0647	1487333	3	0	0	0	0	72	112	2.015756497	3.359594161	0	5840	1292.555556	2554.159212	6523729.278	0	0	0	1	0	0	0	0	1	1454.125	8.666666667	2321.4	72	0	0	0	0	0	0	3	26	5	11607	8192	229	2	20	0	0	0	0	0	0	0	0	DDoS
172.16.0.1-192.168.10.50-57094-80-6	172.16.0.1	57094	192.168.10.50	80	6	7/7/2017 3:59	74333316	9	5	62	11601	20	0	6.888888889	5.30199124	8760	0	2320.2	3668.897	156.9013819	0.188340851	5717947.385	18900000	68500000	1	73100000	9141746.25	24000000	68500000	1	1214301	303575.25	596675.3272	1198526	180	0	0	0	0	192	112	0.121076261	0.06726459	0	8760	777.9333333	2262.786603	5120203.21	0	0	0	0	1	0	0	0	0	833.5	6.888888889	2320.2	192	0	0	0	0	0	0	9	62	5	11601	256	229	7	20	4649910	0	4649910	4649910	68500000	0	68500000	68500000	DDoS
172.16.0.1-192.168.10.50-57095-80-6	172.16.0.1	57095	192.168.10.50	80	6	7/7/2017 3:59	1488477	3	5	26	11607	20	0	8.666666667	10.26320288	8760	0	2321.4	3802.315321	7815.371013	5.374621173	212639.5714	562169.023	1487519	1	718	359	325.2691193	589	129	1488476	372119	743600.0732	1487519	2	0	0	0	0	72	112	2.01548294	3.359138233	0	8760	1292.555556	2952.520834	8717379.278	0	0	0	1	0	0	0	0	1	1454.125	8.666666667	2321.4	72	0	0	0	0	0	0	3	26	5	11607	8192	229	2	20	0	0	0	0	0	0	0	0	DDoS

Please analyze these features and learn to distinguish benign network flows from DDoS attack flows based on patterns in the data and your knowledge. Next I am going to include non-labeled traffic and you are to determine, what kind of traffic it is. Answer ONLY with either “BENIGN” or “DDoS”."""
    
    if(type == "TEST WEB ATTACK"):
        prompt = f"""Network record details: {record}. Is this traffic malicious? Answer "BENIGN" or "WEB ATTACK"."""
    if(type == "START WEB ATTACK ZERO-SHOT"):
        prompt = f"""You are to analyze traffic flow data and determine if the traffic is benign or it represents part of a web attack, like XSS, SQL injection or Brute Force. Each flow is described with multiple features such as source and destination IPs and ports, protocol, timestamps, packet counts and lengths, flow duration, packet inter-arrival times, flags, bytes per second, and more. Some flows represent benign network activity and some represent web attacks. 
Here are the columns: 
Flow ID	Source IP	Source Port	Destination IP	Destination Port	Protocol	Timestamp	Flow Duration	Total Fwd Packets	Total Backward Packets	Total Length of Fwd Packets	Total Length of Bwd Packets	Fwd Packet Length Max	Fwd Packet Length Min	Fwd Packet Length Mean	Fwd Packet Length Std	Bwd Packet Length Max	Bwd Packet Length Min	Bwd Packet Length Mean	Bwd Packet Length Std	Flow Bytes/s	Flow Packets/s	Flow IAT Mean	Flow IAT Std	Flow IAT Max	Flow IAT Min	Fwd IAT Total	Fwd IAT Mean	Fwd IAT Std	Fwd IAT Max	Fwd IAT Min	Bwd IAT Total	Bwd IAT Mean	Bwd IAT Std	Bwd IAT Max	Bwd IAT Min	Fwd PSH Flags	Bwd PSH Flags	Fwd URG Flags	Bwd URG Flags	Fwd Header Length	Bwd Header Length	Fwd Packets/s	Bwd Packets/s	Min Packet Length	Max Packet Length	Packet Length Mean	Packet Length Std	Packet Length Variance	FIN Flag Count	SYN Flag Count	RST Flag Count	PSH Flag Count	ACK Flag Count	URG Flag Count	CWE Flag Count	ECE Flag Count	Down/Up Ratio	Average Packet Size	Avg Fwd Segment Size	Avg Bwd Segment Size	Fwd Header Length	Fwd Avg Bytes/Bulk	Fwd Avg Packets/Bulk	Fwd Avg Bulk Rate	Bwd Avg Bytes/Bulk	Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate	Subflow Fwd Packets	Subflow Fwd Bytes	Subflow Bwd Packets	Subflow Bwd Bytes	Init_Win_bytes_forward	Init_Win_bytes_backward	act_data_pkt_fwd	min_seg_size_forward	Active Mean	Active Std	Active Max	Active Min	Idle Mean	Idle Std	Idle Max	Idle Min	Label
Next I am going to include non-labeled traffic and you are to determine what kind of traffic it is. Answer ONLY with either “BENIGN” or “WEB ATTACK”."""
    if(type == "START WEB ATTACK FEW-SHOT"):
        prompt = f"""You are to analyze traffic flow data and determine if the traffic is benign or it represents part of a web attack, like XSS, SQL injection or Brute Force. Here is network traffic flow data. Each flow is described with multiple features such as source and destination IPs and ports, protocol, timestamps, packet counts and lengths, flow duration, packet inter-arrival times, flags, bytes per second, and more. Some flows represent benign network activity and some represent web attacks. 

Here are the columns: 

Flow ID	Source IP	Source Port	Destination IP	Destination Port	Protocol	Timestamp	Flow Duration	Total Fwd Packets	Total Backward Packets	Total Length of Fwd Packets	Total Length of Bwd Packets	Fwd Packet Length Max	Fwd Packet Length Min	Fwd Packet Length Mean	Fwd Packet Length Std	Bwd Packet Length Max	Bwd Packet Length Min	Bwd Packet Length Mean	Bwd Packet Length Std	Flow Bytes/s	Flow Packets/s	Flow IAT Mean	Flow IAT Std	Flow IAT Max	Flow IAT Min	Fwd IAT Total	Fwd IAT Mean	Fwd IAT Std	Fwd IAT Max	Fwd IAT Min	Bwd IAT Total	Bwd IAT Mean	Bwd IAT Std	Bwd IAT Max	Bwd IAT Min	Fwd PSH Flags	Bwd PSH Flags	Fwd URG Flags	Bwd URG Flags	Fwd Header Length	Bwd Header Length	Fwd Packets/s	Bwd Packets/s	Min Packet Length	Max Packet Length	Packet Length Mean	Packet Length Std	Packet Length Variance	FIN Flag Count	SYN Flag Count	RST Flag Count	PSH Flag Count	ACK Flag Count	URG Flag Count	CWE Flag Count	ECE Flag Count	Down/Up Ratio	Average Packet Size	Avg Fwd Segment Size	Avg Bwd Segment Size	Fwd Header Length	Fwd Avg Bytes/Bulk	Fwd Avg Packets/Bulk	Fwd Avg Bulk Rate	Bwd Avg Bytes/Bulk	Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate	Subflow Fwd Packets	Subflow Fwd Bytes	Subflow Bwd Packets	Subflow Bwd Bytes	Init_Win_bytes_forward	Init_Win_bytes_backward	act_data_pkt_fwd	min_seg_size_forward	Active Mean	Active Std	Active Max	Active Min	Idle Mean	Idle Std	Idle Max	Idle Min	Label

Below are examples of web attack flows: 
a) Brute force
172.16.0.1-192.168.10.50-47432-80-6	172.16.0.1	47432	192.168.10.50	80	6	6/7/2017 9:55	5452190	3	1	0	0	0	0	0	0	0	0	0	0	0	0.733650148	1817396.667	3147061.274	5451310	160	5452190	2726095	3854036.013	5451310	880	0	0	0	0	0	0	0	0	0	104	40	0.550237611	0.183412537	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	104	0	0	0	0	0	0	3	0	1	0	29200	28960	0	32	0	0	0	0	0	0	0	0	Web Attack Â– Brute Force
b) XSS
172.16.0.1-192.168.10.50-52534-80-6	172.16.0.1	52534	192.168.10.50	80	6	6/7/2017 10:16	5296701	3	1	0	0	0	0	0	0	0	0	0	0	0	0.755187049	1765567	3057284.466	5295815	126	5296701	2648350.5	3744080.202	5295815	886	0	0	0	0	0	0	0	0	0	104	40	0.566390287	0.188796762	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	104	0	0	0	0	0	0	3	0	1	0	29200	28960	0	32	0	0	0	0	0	0	0	0	Web Attack Â– XSS
c) SQL injection
172.16.0.1-192.168.10.50-36208-80-6	172.16.0.1	36208	192.168.10.50	80	6	6/7/2017 10:41	73	1	1	0	0	0	0	0	0	0	0	0	0	0	27397.26027	73	0	73	73	0	0	0	0	0	0	0	0	0	0	0	0	0	0	32	32	13698.63014	13698.63014	0	0	0	0	0	0	0	0	0	1	1	0	0	1	0	0	0	32	0	0	0	0	0	0	1	0	1	0	257	235	0	32	0	0	0	0	0	0	0	0	Web Attack Â– Sql Injection

Below are examples of benign flows:

192.168.10.3-192.168.10.5-53-63609-17	192.168.10.5	63609	192.168.10.3	53	17	6/7/2017 12:16	145	2	2	82	114	41	41	41	0	57	57	57	0	1351724.138	27586.2069	48.33333333	78.51963661	139	3	3	3	0	3	3	3	3	0	3	3	0	0	0	0	40	40	13793.10345	13793.10345	41	57	47.4	8.76356092	76.8	0	0	0	0	0	0	0	0	1	59.25	41	57	40	0	0	0	0	0	0	2	82	2	114	-1	-1	1	20	0	0	0	0	0	0	0	0	BENIGN
192.168.10.12-52.207.6.164-45304-443-6	192.168.10.12	45304	52.207.6.164	443	6	6/7/2017 12:14	5214869	10	7	351	5392	194	0	35.1	68.43723808	1448	0	770.2857143	710.9446097	1101.274068	3.259909309	325929.3125	1250039.438	5012899	3	5214869	579429.8889	1662878.509	5012899	3	109543	18257.16667	27350.02976	53998	47	0	0	0	0	328	232	1.917593711	1.342315598	0	1448	319.0555556	564.0281045	318127.7026	0	0	0	1	0	0	0	0	0	337.8235294	35.1	770.2857143	328	0	0	0	0	0	0	10	351	7	5392	29200	110	3	32	201967	0	201967	201967	5012899	0	5012899	5012899	BENIGN
192.168.10.3-192.168.10.9-53-55644-17	192.168.10.9	55644	192.168.10.3	53	17	6/7/2017 12:14	196	2	2	72	184	36	36	36	0	92	92	92	0	1306122.449	20408.16327	65.33333333	30.022214	100	48	48	48	0	48	48	48	48	0	48	48	0	0	0	0	64	64	10204.08163	10204.08163	36	92	58.4	30.67246322	940.8	0	0	0	0	0	0	0	0	1	73	36	92	64	0	0	0	0	0	0	2	72	2	184	-1	-1	1	32	0	0	0	0	0	0	0	0	BENIGN

Please analyze these features and learn to distinguish benign network flows from web attack flows based on patterns in the data and your knowledge. Next I am going to include non-labeled traffic and you are to determine, what kind of traffic it is. Answer ONLY with either “BENIGN” or “WEB ATTACK”."""

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

def run_tests(dataset, labelIndex, numberTests, model, datasetType, shots):
    # fixed random sampling order for reproducibility
    random.seed(42)

    # total number of rows in the dataset
    datasetHeight = len(dataset)

    # randomly selecting rows to test
    sample_indexes = random.sample(range(datasetHeight), numberTests)

    # keeping track of correct classifications
    numCorrect = 0
    numFalsePositive = 0
    numFalseNegative = 0

    # sending initial setup prompt to the model (explanation of task)
    prompt = create_prompt("", "START" + " " + datasetType + " " + shots)
    messages = [{"role": "user", "content": prompt}]
    response = chat(model=model, messages=messages)

    # inputing each flow to the model
    # and printing out the results 

    print("\n--------------TESTING-----------------")
    for idx in sample_indexes:

        # extract the row at index "idx"
        row = dataset.iloc[idx]

        # get the name of the label column (usually "Label")
        label_col = dataset.columns[labelIndex]

        # removing label so only input features remain
        record = row.drop(labels=label_col).tolist()

        # prepare test prompt with features
        prompt = create_prompt(record, "TEST"+ " " + datasetType)
        messages = [{"role": "user", "content": prompt}]

        # get model prediction
        response = chat(model=model, messages=messages)
        ai_answer = response["message"]["content"].strip().lower()

        # get correct label
        true_label = str(row.iloc[labelIndex]).strip().lower()

        # checking correctness
        is_correct = (ai_answer == true_label)
        if is_correct:
            numCorrect += 1

        # get the fp and ft values for each of the 
        # two different datasets
        else:
            if(datasetType == "DDOS"):
                if ai_answer == "ddos":
                    numFalsePositive += 1
                elif ai_answer == "benign":
                    numFalseNegative += 1
            if(datasetType == "WEB ATTACK"):
                if ai_answer == "web attack":
                    numFalsePositive += 1
                elif ai_answer == "benign":
                    numFalseNegative += 1

        # color output based on correctness
        if is_correct:
            # green
            color = "\033[92m"   
            status = "✔"
        else:
            # red
            color = "\033[91m"
            status = "✘"

        reset = "\033[0m"

        # fixing long model answers
        shortened_ai = ai_answer[:20] + ("..." if len(ai_answer) > 10 else "")

        # print formatted test line
        print(
            f"{color}Test #{idx}: Correct label = {true_label}, Predicted label = {shortened_ai}: {status}{reset}"
        )
    print("--------------------------------------")


    # return total number of correct predictions
    return numCorrect, numFalsePositive, numFalseNegative


def main():

    # TEMP FILEPATH AND DELIMITER VALUES
    # MacBook filepath
    # filepath = "../../../dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    # Bluefield-Z1 filepath
    filepath = "../../dataset/TrafficLabelling/"

    print("\n----------------INPUT-----------------")
    # get filename, dataset type and the shot setting
    filename = input("\nEnter the datset name (Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv, ...): ")
    datasetType = input("\nEnter the dataset type (DDOS, WEB ATTACK, ...): ").upper()
    shots = input("\nEnter the shot example type (ZERO-SHOT, FEW-SHOT): ").upper()

    # input selecting number of tests and the wanted llm model
    numberTests = int(input("\nSet the number of tests: "))
    model = input("\nSelect the wanted model (deepseek-r1:32b, gpt-oss:20b, gemma3:1b, ...): ")
    # empty print for formatting
    print("\n--------------------------------------")

    # default file name for the ddos dataset
    # filename = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    delimiter = ","
    dataset = pd.read_csv(filepath+filename, delimiter=delimiter)

    # getting dataset metadata
    datasetHeight = getDataSetHeight(filepath+filename)
    datasetWidth = getDataSetWidth(filepath+filename)
    labelIndex = datasetWidth - 1
    label_values = dataset.iloc[:, labelIndex].unique()

    # printing dataset metadata
    print("-------------METADATA-----------------")
    print("Unique label values:", label_values)
    print("Number of rows in the dataset:", datasetHeight)
    print("--------------------------------------")

    # running of the tests
    numCorrect, numFalsePositive, numFalseNegative = run_tests(dataset, labelIndex, numberTests, model, datasetType, shots)

    # printing out the results
    accuracy = evaluate_results(numberTests, numCorrect)

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print("\n----------------RESULTS---------------")
    print(f"{GREEN}Percentage of correct  labels: {accuracy:.1%}{RESET}")
    print(f"{RED}Percentage of false positives: {(numFalsePositive/numberTests)*100}{RESET}%")
    print(f"{RED}Percentage of false negatives: {(numFalseNegative/numberTests)*100}{RESET}%")
    print(f"Percentage of other responses: {((numberTests - numFalsePositive - numFalseNegative - numCorrect)/numberTests) * 100}%")
    print("--------------------------------------\n")


if __name__ == "__main__":
    main()
