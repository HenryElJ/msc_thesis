# Conversational Explanations using Large Language Models on top of the eXplainable AI Question Bank
## Henry Khalil El-Jawhari (5566377)
### Freie Universität Berlin, Department of Mathematics and Computer Science
### Center for Artificial Intelligence in Public Health Research (ZKI-PH) at Robert Koch Institute (RKI)
##### February 26, 2026

The proliferation and widespread adoption of AI systems into our everyday lives poses a significant risk when these systems behave unexpectedly, or make decisions which unfairly target people/groups of people. AI systems need to be transparent, interpretable, and understandable, in order for them to be trustworthy. Central to achieving this is the field of explainable AI (XAI), where Large Language Models (LLMs) present a valuable opportunity as conversational explanatory agents, transforming complex topics into accessible narratives. These LLMs offer personalised learning, refining the user’s mental models, and filling in knowledge gaps.

In this work, an AI system was created as a prediction tool and explanation interface for vaccination uptake within the healthcare domain. A selection of LLMs (ChatGPT5.2, ClaudeOpus 4.6, Gemini3 Pro) were evaluated on their ability to answer questions from the extended explainable AI question bank (XAI-QB) when provided with this AI system’s source code. Using reference human answers as ground-truth labels, the LLMs were evaluated across the dimensions of accuracy, completeness, and faithfulness, which were used to formulate an overall ACF score. Furthermore, these reference human answers served as a baseline for readability, benchmarking the required level at which technical details could be presented in a comprehensible way.

The result of these evaluations suggests that LLMs can indeed be used as conversational explanatory agents. All LLMs performed comparably well, showcasing their ability to answer questions correctly, covering all of the required topics, in a semantically similar way to reference human answers. ChatGPT5.2 performed the most consistently, ranking first in accuracy and faithfulness. ClaudeOpus 4.6 provided the most complete responses, whilst Gemini3 Pro scored the best for readability. This showcases the relative strengths of different LLMs, suggesting that model selection could be tailored to the use-case and communication needs of different audiences. 

Finally, the ACF score provides insights into where the XAI-QB and AI system can be improved — highlighting questions which need further clarification, and AI system design that lacks transparency.
