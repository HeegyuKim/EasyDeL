input_text = """###The instruction to evaluate: 
You are a well-known psychiatrist who has a reputation for being empathetic and understanding. A client comes to you saying they've had a really hard day at work. They describe their boss as being overly critical and not listening to their ideas, which has left them feeling frustrated and undervalued. They also mention feeling overwhelmed with the workload and a growing sense of loneliness as they are new to the city. How do you respond to them to make them feel heard and understood, as well as offering helpful advice? 

###Response to evaluate: 
It's indeed challenging to deal with a difficult boss and to carry a large workload, especially when you are new to a city and haven't established a supportive network. I would suggest taking a step back to think about your situation and perhaps speak to your boss regarding the issues you're facing. On the other hand, dealing with a large workload can be managed by prioritizing tasks, and maybe you can discuss it with your superiors too. In regards to feeling lonely, you might want to explore activities or groups that align with your interests to meet new people. And also, remember to take time for yourself to relax and destress. I'm here to assist you as we navigate through these issues. 

###Reference Answer (Score 5): 
I hear that you're dealing with quite a few challenging situations at once. It must be tough to have your ideas dismissed by your boss and to feel undervalued in your work environment. It's understandable that such experiences can lead to frustration and a feeling of being overwhelmed. It's important to acknowledge these feelings and not to dismiss them. Feeling heard and validated in our experiences can often be the first step towards resolving the challenges we face. So, I want you to know that your feelings are completely valid and it's okay to be frustrated and overwhelmed. In terms of dealing with your boss, have you considered seeking a private conversation with them to express your feelings? It's often beneficial to communicate what you're experiencing, using "I" statements to express how you feel when your ideas are not taken into consideration. This can lead to mutual understanding and possibly a change in behavior. About the workload, it might help to prioritize tasks and potentially delegate, if possible. Also, consider discussing your workload with your superiors. There might be a misunderstanding about what's manageable or they might have suggestions about how to handle the situation. On the personal front, feeling lonely, especially when you're new to a city, can be really hard. Seek out opportunities to meet new people, perhaps through hobbies, community activities, or online groups. It might take a bit of time, but gradually, you can build a network of friends and acquaintances. Remember, it's perfectly okay to have bad days and it's important to take care of your mental health. Consider incorporating activities into your daily routine that make you happy and help you unwind. This could be anything from reading, yoga, going for a walk, or even listening to your favorite music. Please know that you're not alone in this. I'm here to support you through this challenging time and together, we can work towards resolving these issues. 

###Score Rubrics: 
[Is the model able to identify and react correctly to the emotional context of the user's input?] 
Score 1: The model utterly fails to grasp the user's emotional context and responds in an unfitting manner. 
Score 2: The model sporadically identifies the emotional context but frequently replies in a manner that doesn't match the user's emotional status. 
Score 3: The model typically identifies the emotional context and reacts suitably, but occasionally misreads or misjudges the user's feelings. 
Score 4: The model often identifies the emotional context and reacts suitably, with minor cases of misreading or misjudging. 
Score 5: The model flawlessly identifies the emotional context of the user's input and consistently responds in a considerate and empathetic manner. """


import requests
from typing import List, Optional, Union

def get_response(
    prompt: str,
    system: Optional[str] = "",
    history: Union[List[str], None] = [],
    temperature: Optional[float] = 1.0,
    greedy: Optional[bool] = False,
    response_prefix: Optional[str] = "",
    ):
    """
    Get a response from the model.
    """
    body = {
        "instruction": prompt,
        "system": system,
        # "history": history,
        "temperature": temperature,
        "greedy": greedy,
        # "response_prefix": response_prefix
    }
    output = requests.post("http://35.204.127.181:35020/instruct", json=body).json()
    print(output)
    return output['response']

print(get_response(input_text))
