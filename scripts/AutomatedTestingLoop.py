# AutomatedTestingLoop.py
import subprocess

def get_installed_models():
    # get all locally installed Ollama models
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[1:]
    return [line.split()[0] for line in lines if line.strip()]

def run_automated_tests(dataset, labelIndex, dataset_type, shots, num_tests=20, window_size=5, seed=42):
    # run tests on all installed models and return results
    from ModelTester import run_tests
    import math
    
    results = []
    models = get_installed_models()
    
    print(f"Found {len(models)} installed models")
    
    for model in models:
        print(f"\n{'='*100}")
        print(f"Testing model: {model}")
        print(f"{'='*100}")
        
        try:
            numTP, numTN, numFP, numFN = run_tests(
                dataset, labelIndex, num_tests, model, 
                dataset_type, shots, window_size, seed, False
            )
            
            # calculate metrics
            total = num_tests * window_size
            accuracy = (numTP + numTN) / total
            precision = numTP / (numTP + numFP) if (numTP + numFP) > 0 else 0.0
            recall = numTP / (numTP + numFN) if (numTP + numFN) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # calculate MCC
            numerator = (numTP * numTN) - (numFP * numFN)
            denominator = math.sqrt((numTP + numFP) * (numTP + numFN) * (numTN + numFP) * (numTN + numFN))
            mcc = numerator / denominator if denominator > 0 else 0.0
            
            results.append({
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'tp': numTP,
                'tn': numTN,
                'fp': numFP,
                'fn': numFN
            })
        except Exception as e:
            print(f"Error with {model}: {e}")
            continue
    
    # sort and display results
    if not results:
        print("\nNo results to display.")
        return results
    
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS - SORTED BY F1 SCORE")
    print("="*100)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'MCC':<10}")
    print("-"*100)
    
    for r in results:
        print(f"{r['model']:<25} {r['accuracy']:<12.1%} {r['precision']:<12.1%} "
              f"{r['recall']:<12.1%} {r['f1_score']:<12.1%} {r['mcc']:<10.4f}")
    
    # show best model details
    best = results[0]
    print("\n" + "="*100)
    print("BEST MODEL DETAILS")
    print("="*100)
    print(f"{'Model:':<45} {best['model']}")
    print(f"{'Accuracy:':<45} {best['accuracy']:.1%}")
    print(f"{'Precision:':<45} {best['precision']:.1%}")
    print(f"{'Recall:':<45} {best['recall']:.1%}")
    print(f"{'F1 Score:':<45} {best['f1_score']:.1%}")
    print(f"{'MCC:':<45} {best['mcc']:.4f}")
    print(f"{'Confusion Matrix:':<45} TP={best['tp']}, TN={best['tn']}, FP={best['fp']}, FN={best['fn']}")
    print("="*100)
    
    return results
