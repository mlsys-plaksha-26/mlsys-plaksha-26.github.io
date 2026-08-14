---
title: "Introduction to ML Systems"
author:
    name: Pankaj Pansari
    affiliation: Plaksha University
---

This course is a quantitative, systems-oriented deep dive into LLM inference. *Inference* is the process of giving a ML model an input and obtaining an output.

![ML Inference](llm_inference.png)

We use the term *serving* when talking about inference for several concurrent users. This also refers to a software stack that accepts user requests, schedules them, and executes inference.

## Why Focus on LLM Inference?

There are two reasons motivating our study of the LLM process in depth in our course:

### Scaling of usage

With widespread adoption of AI, the LLM inference workload has become a significant chunk of overall AI compute. There are two factors driving this rapid increase in inference compute:

1. Number of users - growing as more people make increasing use of AI models
2. Inference scaling - our interactions with LLMs these days does not resemble the one prompt, one response patter anymore.

a. Longer - reasoning models generate a lot of reasoning tokens before they answer
b. Wider - on difficult problems, test-time scaling is used. This involves sampling many candidate answers and picking the best one(s) as decided by a verifier
c. Deeper - agentic systems solve a complex task by repeatedly acting, observing, and adapting.

 
### Computational challenges
