"""
Comprehensive Emotion Detection Test Suite
Tests all 28 emotions supported by SamLowe/roberta-base-go_emotions model
"""

import requests
import json
from datetime import datetime

# ML Service endpoint
ML_SERVICE_URL = "https://namanj11-eunoia-ml-service.hf.space/analyze"

# Test cases for all 28 emotions
EMOTION_TEST_CASES = {
    "admiration": {
        "title": "So Impressed",
        "content": "I am in awe of my friend. They just accomplished something incredible that I never thought was possible. I admire their dedication and hard work so much. They are truly inspiring and I look up to them with great respect. What an amazing achievement!",
        "expected": "admiration"
    },
    
    "amusement": {
        "title": "Laughing So Hard",
        "content": "That comedy show was hilarious! I cannot stop laughing. My stomach hurts from laughing so much. That was the funniest thing I have seen in ages. I am still chuckling just thinking about it. It was absolutely entertaining!",
        "expected": "amusement"
    },
    
    "anger": {
        "title": "So Frustrated",
        "content": "I am absolutely furious right now! My colleague took credit for my work in the meeting today. I worked so hard on that project and they just stole it. I am so angry I could scream. This is completely unacceptable and unfair!",
        "expected": "anger"
    },
    
    "annoyance": {
        "title": "Minor Irritation",
        "content": "This is getting on my nerves. The constant noise from construction next door is really bothering me. It is annoying and disruptive. I am getting irritated by all these small inconveniences piling up. It is not the end of the world but it is frustrating.",
        "expected": "annoyance"
    },
    
    "approval": {
        "title": "Good Decision",
        "content": "I agree with the decision made today. It was the right choice and I support it completely. I approve of this direction and think it will lead to positive outcomes. This is a sensible and well-thought-out plan.",
        "expected": "approval"
    },
    
    "caring": {
        "title": "Worried About Mom",
        "content": "My mom has been sick and I am so concerned about her health. I want to take care of her and make sure she gets better. I feel protective and caring towards her. Her wellbeing means everything to me. I hope she recovers soon.",
        "expected": "caring"
    },
    
    "confusion": {
        "title": "So Lost",
        "content": "I do not understand what is happening. Everything is so confusing right now. I feel puzzled and bewildered by this situation. Nothing makes sense anymore. I am completely lost and do not know what to think or do next.",
        "expected": "confusion"
    },
    
    "curiosity": {
        "title": "Want to Learn More",
        "content": "I am so curious about this new technology. I wonder how it works and what makes it tick. I want to explore and discover more about it. There is so much to learn and I am eager to find out everything I can about this fascinating subject.",
        "expected": "curiosity"
    },
    
    "desire": {
        "title": "Really Want This",
        "content": "I desperately want to achieve this goal. I crave success and long for this opportunity. The desire to accomplish this is overwhelming. I yearn for it with all my heart and would do anything to make it happen.",
        "expected": "desire"
    },
    
    "disappointment": {
        "title": "Let Down",
        "content": "I am so disappointed. I did not get the job I interviewed for. I had such high hopes and now I feel let down and discouraged. All that preparation for nothing. This is really disheartening and frustrating to deal with.",
        "expected": "disappointment"
    },
    
    "disapproval": {
        "title": "Not Okay With This",
        "content": "I strongly disagree with this decision. I do not approve of this approach at all. This is wrong and should not be happening. I am against this and think it is a terrible idea that will lead to problems.",
        "expected": "disapproval"
    },
    
    "disgust": {
        "title": "That Was Gross",
        "content": "That was absolutely revolting and disgusting. I feel nauseated just thinking about it. It was repulsive and made me feel sick. I cannot believe how gross that was. It was completely disgusting and unpleasant.",
        "expected": "disgust"
    },
    
    "embarrassment": {
        "title": "So Awkward",
        "content": "I am so embarrassed about what happened today. I tripped and fell in front of everyone at the meeting. My face turned bright red and I felt so awkward and humiliated. I wanted to disappear. It was mortifying and I cannot stop cringing.",
        "expected": "embarrassment"
    },
    
    "excitement": {
        "title": "Best Day Ever",
        "content": "I got promoted at work today! I am so incredibly excited and thrilled. This is exactly what I have been working towards for years. I feel energized and pumped up! I cannot wait to celebrate. This is amazing!",
        "expected": "excitement"
    },
    
    "fear": {
        "title": "So Scared",
        "content": "I am terrified about the medical test results. What if something is seriously wrong? I cannot stop thinking about worst case scenarios. My heart is racing and I feel so anxious and scared about what might happen. I am filled with dread.",
        "expected": "fear"
    },
    
    "gratitude": {
        "title": "So Thankful",
        "content": "I am incredibly grateful for all the help I received. I feel so blessed and thankful for the support from my friends and family. I appreciate everything they have done for me. Their kindness means the world to me. I am truly grateful.",
        "expected": "gratitude"
    },
    
    "grief": {
        "title": "Lost My Father",
        "content": "My father died last month. I am grieving his loss every single day. The pain of losing him is unbearable. I will never see him again. This grief is overwhelming and I feel like a part of me died with him. The funeral was last week and I cannot stop crying.",
        "expected": "grief"
    },
    
    "joy": {
        "title": "Pure Happiness",
        "content": "I am filled with pure joy today. Everything feels wonderful and perfect. My heart is full of happiness and delight. I feel blessed and joyful. Life is beautiful and I am grateful for this moment of pure bliss and contentment.",
        "expected": "joy"
    },
    
    "love": {
        "title": "My Heart is Full",
        "content": "I love my partner so much. They are my everything. When I look at them, my heart fills with warmth and deep affection. I cherish every moment we spend together. They make me feel complete and loved beyond measure. I adore them.",
        "expected": "love"
    },
    
    "nervousness": {
        "title": "Big Presentation",
        "content": "I have a big presentation tomorrow and I am feeling nervous and uneasy. My palms are sweaty and I feel jittery. I am anxious about speaking in front of everyone. What if I mess up? I feel tense and on edge about this.",
        "expected": "nervousness"
    },
    
    "neutral": {
        "title": "Regular Day",
        "content": "Today was a normal day. Nothing particularly good or bad happened. I went about my usual routine and tasks. Everything was average and unremarkable. Just another ordinary day without any strong emotions or events.",
        "expected": "neutral"
    },
    
    "optimism": {
        "title": "Looking Forward",
        "content": "I am feeling hopeful and optimistic about the future. Things are going to get better, I just know it. I believe that good things are coming my way. Tomorrow will be brighter and I am confident everything will work out perfectly.",
        "expected": "optimism"
    },
    
    "pride": {
        "title": "Accomplished Something",
        "content": "I am so proud of myself for completing this difficult project. I worked hard and persevered through challenges. I feel accomplished and satisfied with what I achieved. This is a moment of pride and self-respect. I did it!",
        "expected": "pride"
    },
    
    "realization": {
        "title": "Sudden Understanding",
        "content": "I just realized something important. It suddenly clicked and now I understand what was happening all along. I had an epiphany and everything makes sense now. This realization changes everything. I finally get it!",
        "expected": "realization"
    },
    
    "relief": {
        "title": "Crisis Averted",
        "content": "Thank goodness that is over. I feel so relieved that the situation resolved. The stress and tension have finally lifted. I can breathe again. What a relief! I am glad that difficult period has passed and everything turned out okay.",
        "expected": "relief"
    },
    
    "remorse": {
        "title": "Feeling Guilty",
        "content": "I deeply regret what I said to my friend yesterday. I feel terrible remorse and guilt about hurting their feelings. I wish I could take it back. I am filled with regret and feel bad about my actions. I need to apologize.",
        "expected": "remorse"
    },
    
    "sadness": {
        "title": "Feeling Down",
        "content": "I feel so sad and empty today. Nothing seems to bring me joy anymore. I just want to stay in bed and cry. Everything feels heavy and meaningless. I am overwhelmed with sadness and melancholy.",
        "expected": "sadness"
    },
    
    "surprise": {
        "title": "Unexpected News",
        "content": "Wow! I cannot believe what just happened. My friends threw me a surprise birthday party! I had no idea they were planning this. I am completely shocked and amazed. I never saw this coming at all! What a surprise!",
        "expected": "surprise"
    }
}


def test_emotion(emotion_name, test_case):
    try:
        response = requests.post(
            ML_SERVICE_URL,
            json={
                "title": test_case["title"],
                "content": test_case["content"]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result.get("data", {})
                detected = data.get("primary_emotion")
                confidence = data.get("emotion_confidence", 0)
                expected = test_case["expected"]
                
                # Check if detection matches expected
                is_correct = detected == expected
                status = "✅ PASS" if is_correct else "❌ FAIL"
                
                return {
                    "emotion": emotion_name,
                    "expected": expected,
                    "detected": detected,
                    "confidence": confidence,
                    "is_correct": is_correct,
                    "status": status,
                    "all_emotions": data.get("detected_emotions", [])
                }
            else:
                return {
                    "emotion": emotion_name,
                    "status": "❌ ERROR",
                    "error": result.get("error", "Unknown error")
                }
        else:
            return {
                "emotion": emotion_name,
                "status": "❌ ERROR",
                "error": f"HTTP {response.status_code}"
            }
            
    except Exception as e:
        return {
            "emotion": emotion_name,
            "status": "❌ ERROR",
            "error": str(e)
        }


def run_all_tests():
    print(f"Testing {len(EMOTION_TEST_CASES)} emotions")
    print(f"ML Service: {ML_SERVICE_URL}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    correct_count = 0
    total_count = 0
    
    for emotion_name, test_case in EMOTION_TEST_CASES.items():
        print(f"Testing {emotion_name.upper()}...", end=" ")
        result = test_emotion(emotion_name, test_case)
        results.append(result)
        
        if result.get("is_correct"):
            correct_count += 1
            print(f"{result['status']} - Detected: {result['detected']} ({result['confidence']:.2%})")
        elif "error" in result:
            print(f"{result['status']} - {result['error']}")
        else:
            print(f"{result['status']} - Expected: {result['expected']}, Got: {result['detected']} ({result['confidence']:.2%})")
        
        total_count += 1
    
    print()
    print("TEST RESULTS SUMMARY")
    
    # Calculate statistics
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Total Tests: {total_count}")
    print(f"Passed: {correct_count}")
    print(f"Failed: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()
    
    # Show failed tests
    failed_tests = [r for r in results if not r.get("is_correct") and "error" not in r]
    if failed_tests:
        print("FAILED DETECTIONS:")
        for test in failed_tests:
            print(f"  {test['emotion'].upper()}")
            print(f"    Expected: {test['expected']}")
            print(f"    Detected: {test['detected']} ({test['confidence']:.2%})")
            if test.get('all_emotions'):
                print(f"    Top 3 emotions detected:")
                for emotion in test['all_emotions'][:3]:
                    print(f"      - {emotion['emotion']}: {emotion['score']:.2%}")
            print()
    
    # Show successful tests
    passed_tests = [r for r in results if r.get("is_correct")]
    if passed_tests:
        print("SUCCESSFUL DETECTIONS:")
        avg_confidence = sum(t['confidence'] for t in passed_tests) / len(passed_tests)
        print(f"Average Confidence: {avg_confidence:.2%}")
        print()
        for test in passed_tests:
            print(f"  ✅ {test['emotion'].upper()}: {test['confidence']:.2%}")
    
    # Show errors
    error_tests = [r for r in results if "error" in r]
    if error_tests:
        print()
        print("ERRORS:")
        for test in error_tests:
            print(f"  ❌ {test['emotion'].upper()}: {test['error']}")
    
    # Save results to JSON file
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_count,
            "passed": correct_count,
            "failed": total_count - correct_count,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_all_tests()
