# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Final Project,

Part 2/3: Design Write up and Exploratory Data Analysis

#### Design Write up
Project outlines are a valuable resource when working with data projects, as they help keep your project organized.  A well constructed outline can clarify your goals and serve as a checklist when conducting research and analysis.

For this project, you will need to complete a problem statement and research design outline for one of the three lightning talks you designed during pt. 1. This will serve as the starting point for your analysis. Make sure to include a specific aim and hypthesis, well-defined risks and assumptions, and clearly articulated goals and success metrics.

Remember, completing this task earlier will give you more chances to iterate and improve!


#### Exploratory Data Analysis

Exploratory data analysis is a crucial and informative step in the data process. It helps confirm or deny your initial hypotheses and helps visualize the relationships among your data. Your exploratory analysis also informs the kinds of data transformations that you'll need to optimize for machine learning models.

In this assignment, you will explore and visualize your initial analysis in order to effectively tell your data's story. You'll create an iPython notebook that explores your data mathematically, using a python visualization package.

# Project Design Writeup and Approval Template

Follow this as a guide to completing the project design writeup. The questions for each section are merely there to suggest what the baseline should cover; be sure to use detail as it will make the project much easier to approach as the class moves on.

### Project Problem and Hypothesis
* What's the project about? What problem are you solving?
* Where does this seem to reside as a machine learning problem? Are you predicting some continuous number, or predicting a binary value?
* What kind of impact do you think it could have?
* What do you think will have the most impact in predicting the value you are interested in solving for?

### Datasets
* Description of data set available, at the field level (see table)
* If from an API, include a sample return (this is usually included in API documentation!) (if doing this in markdown, use the javacription code tag)

### Domain knowledge
* What experience do you already have around this area?
* Does it relate or help inform the project in any way?
* What other research efforts exist?
    * Use a quick Google search to see what approaches others have made, or talk with your colleagues if it is work related about previous attempts at similar problems.
    * This could even just be something like "the marketing team put together a forecast in excel that doesn't do well."
    * Include a benchmark, how other models have performed, even if you are unsure what the metric means.

### Project Concerns
* What questions do you have about your project? What are you not sure you quite yet understand? (The more honest you are about this, the easier your instructors can help).
* What are the assumptions and caveats to the problem?
    * What data do you not have access to but wish you had?
    * What is already implied about the observations in your data set? For example, if your primary data set is twitter data, it may not be representative of the whole sample (say, predicting who would win an election)
* What are the risks to the project?
    * What's the cost of your model being wrong? (What's the benefit of your model being right?)
    * Is any of the data incorrect? Could it be incorrect?

### Outcomes
* What do you expect the output to look like?
* What does your target audience expect the output to look like?
* What gain do you expect from your most important feature on its own?
* How complicated does your model have to be?
* How successful does your project have to be in order to be considered a "success"?
* What will you do if the project is a bust (this happens! but it shouldn't here)?



### DELIVERABLES

#### Project Design Writeup

- **Requirements:**
    - Well-articulated problem statement with "specific aim" and hypothesis, based on your lightning talk
    - An outline of any potential methods and models
    - Detailed explanation of extant data available (ie: build a data dictionary or link to pre-built data dictionaries)
    - Describe any outstanding questions, assumptions, risks, caveats
    - Demonstrate domain knowledge, including specific features or relevant benchmarks from similar projects
    - Define your goals and criteria, in order to explain what success looks like

- **Bonus:**
    - Consider alternative hypotheses: if your project is a regression problem, is it possible to rewrite it as a classification problem?
    - "Convert" your goal metric from a statistical one (like Mean Squared Error) and tie it to something non-data people can understand, like a cost/benefit analysis, etc.

#### Exploratory Analysis Writeup

- **Requirements:**
   * Review the data set and project with an EIR during office hours.
   * Practice importing (potentially unformatted) data into clean matrices|data frames, and if necessary, export into a form that makes sense (text files or a database, for example).
   * Explore the mathematical properties and visualize data through a python visualization tool (matplotlib and seaborn)
   * Provide insight about the data set and any impact on a hypothesis.

- **Detailed Breakdown:**
   * A well organized iPython notebook with code and output
   * At least one visual for each independent variable and, if possible, its relationship to your dependent variable.
      * It's just as important to show what's not correlated as it is to show any actual correlations found.
      * Visuals should be well labeled and intuitive based on the data types.
        * For example, if your X variable is temperature and Y is "did it rain," a reasonable visual would be two histograms of temperature, one where it rained, and one where it didn't.
      * Tables are a perfectly valid visualization tool! Interweave them into your work.

- **Bonus:**
   - Surface and share your analysis online. Jupyter makes this very simple and the setup should not take long.
   - Try experimenting with other visualization languages; python/pandas-highcharts, shiny/r, or for a real challenge, d3 on its own. Interactive data analysis opens the doors for others to easily interpret your work and explore the data themselves!


### RESOURCES

#### Suggestions for Getting Started

- Keep the project simple! The "cool" part of the analysis will come; just looking at simple relationships between variables can be incredibly insightful.
- Consider building some helper functions that help you quickly visualize and interpret data.
   - Exploratory data analysis should be formulaic; the code should not be holding you back. There are plenty of "starter code" examples from class materials.
- **DRY:** Don't Repeat Yourself! If you see yourself copy and pasting code a lot, turn it into a function, and use the function instead!



