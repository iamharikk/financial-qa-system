import torch
import time
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import SequenceMatcher
import re

class BaselineBenchmark:
    def __init__(self, model_name="gpt2"):
        """Initialize the baseline benchmark with pre-trained model"""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Test questions (first 12 from the dataset)
        self.test_questions = [
            {
                "question": "What was TCS's sales turnover in Mar '25?",
                "expected_answer": "TCS's sales turnover in Mar '25 was Rs. 214,853.00 crores."
            },
            {
                "question": "What was TCS's net profit in Mar '25?",
                "expected_answer": "TCS's reported net profit in Mar '25 was Rs. 48,057.00 crores."
            },
            {
                "question": "What was TCS's employee cost in Mar '25?", 
                "expected_answer": "TCS's employee cost in Mar '25 was Rs. 107,300.00 crores."
            },
            {
                "question": "What was TCS's total income in Mar '24?",
                "expected_answer": "TCS's total income in Mar '24 was Rs. 208,627.00 crores."
            },
            {
                "question": "What was TCS's book value per share in Mar '25?",
                "expected_answer": "TCS's book value per share in Mar '25 was Rs. 209.00."
            },
            {
                "question": "What was TCS's cash and bank balance in Mar '25?",
                "expected_answer": "TCS's cash and bank balance in Mar '25 was Rs. 7,152.00 crores."
            },
            {
                "question": "What was TCS's operating profit in Mar '24?",
                "expected_answer": "TCS's operating profit in Mar '24 was Rs. 55,847.00 crores."
            },
            {
                "question": "What were TCS's total expenses in Mar '25?",
                "expected_answer": "TCS's total expenses in Mar '25 were Rs. 156,924.00 crores."
            },
            {
                "question": "What was TCS's earnings per share in Mar '24?",
                "expected_answer": "TCS's earnings per share in Mar '24 was Rs. 120.39."
            },
            {
                "question": "What was TCS's networth in Mar '25?",
                "expected_answer": "TCS's networth in Mar '25 was Rs. 75,617.00 crores."
            },
            {
                "question": "What was TCS's depreciation in Mar '25?",
                "expected_answer": "TCS's depreciation in Mar '25 was Rs. 4,220.00 crores."
            },
            {
                "question": "What was TCS's tax expense in Mar '24?",
                "expected_answer": "TCS's tax expense in Mar '24 was Rs. 14,043.00 crores."
            }
        ]
    
    def generate_response(self, question, max_length=100, temperature=0.7):
        """Generate response for a given question"""
        # Format the input prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Measure inference time
        start_time = time.time()
        
        # Generate response with confidence scores
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Decode response
        generated_ids = outputs.sequences[0]
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract only the answer part (everything after "Answer:")
        answer_start = full_response.find("Answer:") + len("Answer:")
        generated_answer = full_response[answer_start:].strip()
        
        # Calculate confidence score (average of token probabilities)
        if hasattr(outputs, 'scores') and outputs.scores:
            # Get probabilities for generated tokens
            probs = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(probs, dim=-1)
            
            # Get probability of each generated token
            generated_tokens = generated_ids[input_ids.shape[1]:]
            token_probs = []
            
            for i, token_id in enumerate(generated_tokens):
                if i < len(outputs.scores):
                    prob = probs[0, i, token_id].item()
                    token_probs.append(prob)
            
            confidence = np.mean(token_probs) if token_probs else 0.0
        else:
            confidence = 0.0
        
        return generated_answer, confidence, inference_time
    
    def calculate_similarity(self, generated, expected):
        """Calculate similarity between generated and expected answers"""
        # Clean and normalize text
        generated_clean = re.sub(r'[^\w\s.]', '', generated.lower())
        expected_clean = re.sub(r'[^\w\s.]', '', expected.lower())
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, generated_clean, expected_clean).ratio()
        return similarity
    
    def extract_numerical_values(self, text):
        """Extract numerical values from text for comparison"""
        # Pattern to match numbers with crores, lakhs, etc.
        pattern = r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crores?|lakhs?)?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Also match standalone numbers
        number_pattern = r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b'
        number_matches = re.findall(number_pattern, text)
        
        all_numbers = matches + number_matches
        return [num.replace(',', '') for num in all_numbers]
    
    def evaluate_accuracy(self, generated, expected):
        """Evaluate accuracy based on similarity and numerical values"""
        # Text similarity
        text_similarity = self.calculate_similarity(generated, expected)
        
        # Check if key numerical values match
        gen_numbers = self.extract_numerical_values(generated)
        exp_numbers = self.extract_numerical_values(expected)
        
        numerical_match = False
        if gen_numbers and exp_numbers:
            # Check if any generated number matches expected numbers
            for gen_num in gen_numbers:
                for exp_num in exp_numbers:
                    try:
                        if abs(float(gen_num) - float(exp_num)) < 0.01:
                            numerical_match = True
                            break
                    except ValueError:
                        continue
                if numerical_match:
                    break
        
        # Combined accuracy score
        if numerical_match:
            accuracy = max(0.8, text_similarity)  # High score if numbers match
        else:
            accuracy = text_similarity
        
        return accuracy
    
    def run_benchmark(self):
        """Run the complete benchmark evaluation"""
        print("Starting Baseline Benchmark Evaluation")
        print("=" * 60)
        
        results = []
        total_inference_time = 0
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"\nTest {i}/{len(self.test_questions)}")
            print(f"Question: {test_case['question']}")
            
            # Generate response
            generated_answer, confidence, inference_time = self.generate_response(
                test_case['question']
            )
            
            # Calculate accuracy
            accuracy = self.evaluate_accuracy(generated_answer, test_case['expected_answer'])
            
            # Store results
            result = {
                'test_id': i,
                'question': test_case['question'],
                'expected_answer': test_case['expected_answer'],
                'generated_answer': generated_answer,
                'accuracy': accuracy,
                'confidence': confidence,
                'inference_time': inference_time
            }
            results.append(result)
            total_inference_time += inference_time
            
            print(f"Expected: {test_case['expected_answer']}")
            print(f"Generated: {generated_answer}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Inference Time: {inference_time:.3f}s")
            print("-" * 40)
        
        # Calculate overall statistics
        accuracies = [r['accuracy'] for r in results]
        confidences = [r['confidence'] for r in results]
        inference_times = [r['inference_time'] for r in results]
        
        overall_stats = {
            'total_questions': len(self.test_questions),
            'average_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'average_inference_time': np.mean(inference_times),
            'total_inference_time': total_inference_time,
            'questions_above_50_accuracy': sum(1 for acc in accuracies if acc > 0.5),
            'questions_above_80_accuracy': sum(1 for acc in accuracies if acc > 0.8)
        }
        
        # Print overall results
        print("\n" + "=" * 60)
        print("BASELINE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Questions Evaluated: {overall_stats['total_questions']}")
        print(f"Average Accuracy: {overall_stats['average_accuracy']:.3f} ± {overall_stats['accuracy_std']:.3f}")
        print(f"Average Confidence: {overall_stats['average_confidence']:.3f} ± {overall_stats['confidence_std']:.3f}")
        print(f"Average Inference Time: {overall_stats['average_inference_time']:.3f}s")
        print(f"Total Inference Time: {overall_stats['total_inference_time']:.3f}s")
        print(f"Questions with >50% Accuracy: {overall_stats['questions_above_50_accuracy']}/{overall_stats['total_questions']}")
        print(f"Questions with >80% Accuracy: {overall_stats['questions_above_80_accuracy']}/{overall_stats['total_questions']}")
        
        # Save results to file
        benchmark_results = {
            'model': 'gpt2-baseline',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_stats': overall_stats,
            'detailed_results': results
        }
        
        with open('baseline_benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\nDetailed results saved to: baseline_benchmark_results.json")
        
        return benchmark_results

def main():
    # Run baseline benchmark
    benchmark = BaselineBenchmark()
    results = benchmark.run_benchmark()

if __name__ == "__main__":
    main()