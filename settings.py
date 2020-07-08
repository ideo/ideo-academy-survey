import pathlib

DATA_PATH = pathlib.Path("dropbox/data")
BASE_CSV_PATH = DATA_PATH / "valid_responses_clustered_by_purchase_management.csv"
CRUNCHBASE_2015_PATH = DATA_PATH / "crunchbase2015"
CRUNCHBASE_2020_PATH = DATA_PATH / "crunchbase2020"

QUALTRICS_DROP_COLS = [
    "Response Type",
    "Start Date",
    "End Date",
    "IP Address",
    "Duration (in seconds)", #open to revisiting this
    "Recorded Date",
    "Response ID",
    "Recipient Last Name",
    "Recipient First Name",
    "Recipient Email",
    "External Data Reference",
    "Location Latitude",
    "Location Longitude",
    "Distribution Channel",
    "User Language",
    "PID",
    "psid"
]


SKIP_COLS = [
    "Unnamed: 0", 
    "Progress",
    "Finished",
    "What best describes your employment type?",
    "What is your biggest challenge when it comes to building your team(s) capabilities?", #can come back
    "How interested would you be in this type of solution as a whole?",
    "Do you feel you would be able to easily find this solution elsewhere?",
    "Do you currently have a similar solution in your organization?",
    "What is the name of the solution you currently use?",
    "Would you invest in the next year in a solution with the following benefits? - Easily communicating a new process to leadership",
    "Would you invest in the next year in a solution with the following benefits? - Setting up a new process for the team to lean on",
    "Would you invest in the next year in a solution with the following benefits? - Getting the process started fast",
    "Would you invest in the next year in a solution with the following benefits? - Consistently delivering value to customers",
    "Would you invest in the next year in a solution with the following benefits? - Focusing the team with a concrete time bound plan",
    "Would you invest in the next year in a solution with the following benefits? - Solving hard challenges creatively",
    "Would you invest in the next year in a solution with the following benefits? - Assessing teams’ strengths and growth areas",
    "Would you invest in the next year in a solution with the following benefits? - Receiving recommendations for tools and methods",
    "Would you invest in the next year in a solution with the following benefits? - Seeing how teams are collaborating",
    "Would you invest in the next year in a solution with the following benefits? - Learning new skills on the job",
    "Would you invest in the next year in a solution with the following benefits? - Scaling the process across the organization",
    'What type of budget might you use for a solution similar to that of the concept "Learn While Doing?" - Other - Text',
    "Approximately how many people are on those teams you selected above (combined)?",
    "At what per-seat, per month price (in USD) would you be willing to spend to have access to this tool?",
    "At what per-seat, per month price (in USD) would this tool be too expensive?",
    "At what per-seat, per month price (in USD) would you consider the price to be so low that you would be concerned about the quality of the product?",
    "Is there anything you’d change about this concept to make it right for your team? If so, what?",
    "[Optional] What company do you work for?",
    "Which of the following best describes the industry that your organization falls under? - Selected Choice",
    "Which of the following best describes the industry that your organization falls under? - Other - Text",
    "Which of the following best describes your current job function? - Selected Choice",
    "Which of the following best describes your current job function? - Other - Text"
]

#skip for now
MULTISELECT_COLS = [
    "Which of the following business approaches do you use (or aspire to use) at your organization?",
    "What type of impact are you are trying to achieve in your work? Choose your top 3.",
    "What skill building topics are most relevant to your team(s) today? Select the top 3. - Selected Choice",
    "What skill building topics are most relevant to your team(s) today? Select the top 3. - Other - Text",
    "Here is a list of benefits this concept is supposed to deliver. Please choose the 3 most important benefits to you.",
    'What level within your organization is the concept "Learn While Doing" for? Check all that apply.',
    'Which teams in your organization do you see the concept "Learn While Doing" being most relevant for? Check all that apply'
]

#skip for now
AGREE_DISAGREE_COLS = [
    "Evaluate your organization in terms of the following statements: - We are an established company",
    "Evaluate your organization in terms of the following statements: - We are a new venture",
    "Evaluate your organization in terms of the following statements: - We were recently acquired",
    "Evaluate your organization in terms of the following statements: - We had a recent change in leadership",
    "Evaluate your organization in terms of the following statements: - We are in the middle of a significant transformation",
    "Evaluate your organization in terms of the following statements: - We have people in charge of scaling new capabilities internally (e.g. Agile)",
    "Evaluate your organization in terms of the following statements: - We are experiencing high employee turnover",
    "Evaluate your organization in terms of the following statements: - We are trying to keep up with the times",
    "Evaluate your organization in terms of the following statements: - We buy software to solve problems",
    "Evaluate your organization in terms of the following statements: - We hire people to solve problems",
]

#skip for now
IMPORTANCE_COLS = [
    "How important are the following additional features to you? - Modules for training internal facilitators",
    "How important are the following additional features to you? - Network of external certified coaches for hire",
    "How important are the following additional features to you? - Vision-setting coached session with leadership",
    "How important are the following additional features to you? - Individual coaching"
]


STANDARD_DUMMY_COLS = {
    "Which business function best describes your core responsibility?":"empFunc",
    "What best describes your company size?":"bizSize",
    "What best describes your role in building capabilities within your organization?  Select the most relevant.": "buildCapability",
    "Which of the following statements best describes your company’s strategy today?":"bizStrategy",
    'What type of budget might you use for a solution similar to that of the concept "Learn While Doing?" - Selected Choice':"budgetCategory",
    'What would best describe your involvement in the purchase process of a solution similar to that of the concept "Learn While Doing?"':"purchaseRole",
    "What is your age?":"empAge",
    "Which statement best describes your current management situation?": "manageSituation",
    "What best describes your highest level of education?":"edLevel",
    "What is your base salary range before bonus or taxes?":"salaryRange",
    "How many people directly report to you?":"directReports",
    "What region do you live in?":"region"
}

#target columns
UTILITY_SCORE_COLS = [
 "Equip your workforce with modern skills while solving your organization's top strategic challenges",
 "Build new capabilities while doing the work that matters to your team",
 "Hone your team's ability to continually experiment and build on the resulting knowledge",
 "Equip teams with the tools and mindset to solve problems independently",
 "Stay ahead of industry disruption by fostering an agile, resilient organizational culture",
 "Lead your organization through a digital transformation",
 "Cultivate shared processes and mindsets so your team can achieve better results",
 "Develop customer-centered skills to repeatedly build products, services, and experiences that appeal to your target audience",
 "Help employees develop customer-centered skills while working towards company strategy",
 "Scale new processes and methodologies across your organization with tools every team will find valuable"
]
