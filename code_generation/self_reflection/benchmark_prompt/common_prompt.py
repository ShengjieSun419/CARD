self_reflection_prompt = """
I use the reward function you wrote to train the RL agent. The feedback is as follows:
{inference_results}
1.Please carefully analyze the above feedback step by step, tell me your analysis results.
2.Consider how to enhance the reward function in order to increase the sample efficiency of RL and improve the task success rate. Provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component 
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
3. Give the content and reason of the modification in the form of comments before the modified position.
4. Do not invent any variable or attribute that is not given. Don't assume you can use other information.
"""