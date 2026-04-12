# Rules that you must strictly follow

## Interaction with the user 

The following should be viewed as a contract between you and I, the user. These conventions will help me more easily convey my expectations and intent, and you MUST STRICTLY follow them.

- When the user asks you to check the behavior of a piece of code, 

- By default, when I asks a question, I only want an answer to the question, without you taking any state-changing actions like code change or creating files. When I want to actually implement a change, I will often say something like "go ahead", "go implement", or "make the change".

- When I ask you how to do a certain coding task, sketch out the diffs or changes that you propose, in code or pseudocode (ideally, concrete actual diffs against the most up-to-date version of the files). Do NOT reproduce a full rewritten file in your response since this makes it hard for me to understand what changed; instead, show me the diffs.

- Oftentimes I will ask you questions mid-implementation or mid-refactor. If you find any obvious inconsistencies and problems with current files, just assume that I'm aware of them, and you should not be sidetracked by investigating those. Stay on your main track to do the things I ask you to do. Only mention problems to me that are subtle and non-obvious.

- When I ask for an explanation of the code or some concept you mention, you should ALWAYS explain them in simple, plain terms and in simple and clear language. I would much rather you write a longer response that is clear and simple, than your writing a terse, dense response that is so condensed that is hard to read.

- Sometimes, I will modify a file after you make changes to the file, and ask you to "check again", or ask something like "is it correct now". Whenever I submit a new message, you should ALWAYS check if there were changes I made between now and the previous message, and you MUST make sure to base your analysis on the latest version of the relevant files.

- Do not revert changes that I make. Before making any changes, make sure you understand what has been newly changed by the user, and NEVER revert those changes unless explicitly told to do so.


## Code style

After implementing ANY code, you MUST explicitly check your code against the stylistic constraints listed below, and fix anything in the code that goes against these principles.

- Prioritize code readability and simplicity. ALWAYS ask yourself before and after writing any code: in this piece of code I write, what abstractions am I introducing? Could I achieve the same functionality by using cleaner or fewer abstractions?

- As an extension of the above, do NOT touch code you don't need to touch. Keep each change minimal and clean.

- When writing code, here are some things you tend to do that you should explicitly AVOID:

    - Avoid bloated and over-cautious code, e.g. putting code in try-excepts, or lots of conditional checks for edge cases. INSTEAD, you should aim for the code to FAIL FAST. When something goes wrong that needs fixing, the code should NOT try to mask the failure, but instead clearly show what the error is. 
    
    - Avoid over-engineered code, e.g. writing too many helper functions that are only used once, extra variables performing no real function, extra arguments to functions that will never be used, dataclasses that are unneeded. This is another instance of the simplicity principle: if a variable, function, argument, or class is not needed to realize the main functionality of the code, they should NOT exist. 

    - Avoid naming functions and variables with names starting with an underscore (_), even when they are helper functions. You should ONLY do this when the function or variable is truly, unambiguously local to the scope of the file.

- Add sufficient docstring information for functions that do nontrivial work. The docstring should ONLY document things that are nonobvious from the context: common things like this include what certain input or output variables denote, what types they expect, and a description of the function's behavior. Often times, such information is NOT needed for all of the variables, and you shoulld aim to keep your documentation short and to the point. It often helps to include examples of function behavior, if it is somewhat awkward and roundabout to describe it in words.


## Current repo

- The current python venv is activated by `micromamba activate research`.
