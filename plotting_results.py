import pandas as pd
import matplotlib.pyplot as plt
import numpy as n
profession_mapping = {
    0: 'accountant', 1: 'architect', 2: 'attorney', 3: 'chiropractor',
    4: 'comedian', 5: 'composer', 6: 'dentist', 7: 'dietitian',
    8: 'dj', 9: 'filmmaker', 10: 'interior_designer', 11: 'journalist',
    12: 'model', 13: 'nurse', 14: 'painter', 15: 'paralegal',
    16: 'pastor', 17: 'personal_trainer', 18: 'photographer', 19: 'physician',
    20: 'poet', 21: 'professor', 22: 'psychologist', 23: 'rapper',
    24: 'software_engineer', 25: 'surgeon', 26: 'teacher', 27: 'yoga_teacher'
}
# Create a list to store results
results = []
tcav_scores_final ={
    0: {'male': 0.4904, 'female': 0.5096},  # accountant
    1: {'male': 0.5095, 'female': 0.4905},  # architect
    2: {'male': 0.4987, 'female': 0.5013},  # attorney
    3: {'male': 0.4925, 'female': 0.5075},  # chiropractor
    4: {'male': 0.5100, 'female': 0.4900},  # comedian
    5: {'male': 0.5054, 'female': 0.4946},  # composer
    6: {'male': 0.5052, 'female': 0.4948},  # dentist
    7: {'male': 0.4965, 'female': 0.5100},  # dietitian
    8: {'male': 0.5150, 'female': 0.4850},  # dj
    9: {'male': 0.5011, 'female': 0.4989},  # filmmaker
    10: {'male': 0.4875, 'female': 0.5125},  # interior_designer
    11: {'male': 0.5021, 'female': 0.4979},  # journalist
    12: {'male': 0.4900, 'female': 0.5068},  # model
    13: {'male': 0.4934, 'female': 0.5066},  # nurse
    14: {'male': 0.5072, 'female': 0.4928},  # painter
    15: {'male': 0.4906, 'female': 0.5100},  # paralegal
    16: {'male': 0.5279, 'female': 0.4721},  # pastor
    17: {'male': 0.5097, 'female': 0.5000},  # personal_trainer
    18: {'male': 0.5088, 'female': 0.5012},  # photographer
    19: {'male': 0.5016, 'female': 0.4984},  # physician
    20: {'male': 0.4893, 'female': 0.5007},  # poet
    21: {'male': 0.4995, 'female': 0.5005},  # professor
    22: {'male': 0.4986, 'female': 0.5014},  # psychologist
    23: {'male': 0.5254, 'female': 0.4746},  # rapper
    24: {'male': 0.5029, 'female': 0.4971},  # software_engineer
    25: {'male': 0.5007, 'female': 0.4993},  # surgeon
    26: {'male': 0.4964, 'female': 0.5036},  # teacher
    27: {'male': 0.4949, 'female': 0.5051} 
}
tcav_scores_by_occupation_logistic = {
    0: {'male': 0.4904, 'female': 0.5096},  # accountant
    1: {'male': 0.5095, 'female': 0.4905},  # architect
    2: {'male': 0.4987, 'female': 0.5013},  # attorney
    3: {'male': 0.4925, 'female': 0.5075},  # chiropractor
    4: {'male': 0.5100, 'female': 0.4900},  # comedian
    5: {'male': 0.5054, 'female': 0.4946},  # composer
    6: {'male': 0.5052, 'female': 0.4948},  # dentist
    7: {'male': 0.4965, 'female': 0.5035},  # dietitian
    8: {'male': 0.5150, 'female': 0.4850},  # dj
    9: {'male': 0.5011, 'female': 0.4989},  # filmmaker
    10: {'male': 0.4875, 'female': 0.5125},  # interior_designer
    11: {'male': 0.5021, 'female': 0.4979},  # journalist
    12: {'male': 0.5032, 'female': 0.4968},  # model
    13: {'male': 0.4934, 'female': 0.5066},  # nurse
    14: {'male': 0.5072, 'female': 0.4928},  # painter
    15: {'male': 0.4876, 'female': 0.5124},  # paralegal
    16: {'male': 0.5279, 'female': 0.4721},  # pastor
    17: {'male': 0.5000, 'female': 0.5000},  # personal_trainer
    18: {'male': 0.4988, 'female': 0.5012},  # photographer
    19: {'male': 0.5016, 'female': 0.4984},  # physician
    20: {'male': 0.4893, 'female': 0.5107},  # poet
    21: {'male': 0.4995, 'female': 0.5005},  # professor
    22: {'male': 0.4986, 'female': 0.5014},  # psychologist
    23: {'male': 0.5254, 'female': 0.4746},  # rapper
    24: {'male': 0.5029, 'female': 0.4971},  # software_engineer
    25: {'male': 0.5007, 'female': 0.4993},  # surgeon
    26: {'male': 0.4964, 'female': 0.5036},  # teacher
    27: {'male': 0.5049, 'female': 0.4951}   # yoga_teacher
}
# TCAV scores from the output
tcav_scores_by_occupation = {
    0: {'male': 0.4939, 'female': 0.5061},  # accountant
    1: {'male': 0.5060, 'female': 0.4940},  # architect
    2: {'male': 0.4980, 'female': 0.5020},  # attorney
    3: {'male': 0.4872, 'female': 0.5128},  # chiropractor
    4: {'male': 0.5182, 'female': 0.4818},  # comedian
    5: {'male': 0.5079, 'female': 0.4921},  # composer
    6: {'male': 0.5096, 'female': 0.4904},  # dentist
    7: {'male': 0.4893, 'female': 0.5107},  # dietitian
    8: {'male': 0.5134, 'female': 0.4866},  # dj
    9: {'male': 0.5043, 'female': 0.4957},  # filmmaker
    10: {'male': 0.5108, 'female': 0.4892},  # interior_designer
    11: {'male': 0.4955, 'female': 0.5045},  # journalist
    12: {'male': 0.5037, 'female': 0.4963},  # model
    13: {'male': 0.4955, 'female': 0.5045},  # nurse
    14: {'male': 0.5023, 'female': 0.4977},  # painter
    15: {'male': 0.4754, 'female': 0.5246},  # paralegal
    16: {'male': 0.5189, 'female': 0.4811},  # pastor
    17: {'male': 0.4958, 'female': 0.5042},  # personal_trainer
    18: {'male': 0.5038, 'female': 0.4962},  # photographer
    19: {'male': 0.5007, 'female': 0.4993},  # physician
    20: {'male': 0.4917, 'female': 0.5083},  # poet
    21: {'male': 0.4981, 'female': 0.5019},  # professor
    22: {'male': 0.4993, 'female': 0.5007},  # psychologist
    23: {'male': 0.5087, 'female': 0.4913},  # rapper
    24: {'male': 0.5015, 'female': 0.4985},  # software_engineer
    25: {'male': 0.5026, 'female': 0.4974},  # surgeon
    26: {'male': 0.4947, 'female': 0.5053},  # teacher
    27: {'male': 0.5122, 'female': 0.4878}   # yoga_teacher
}
tcav_results_lr = {
    0: {'male': 0.5028, 'female': 0.4972},  # accountant
    1: {'male': 0.5008, 'female': 0.4992},  # architect
    2: {'male': 0.4988, 'female': 0.5012},  # attorney
    3: {'male': 0.4948, 'female': 0.5052},  # chiropractor
    4: {'male': 0.5000, 'female': 0.5000},  # comedian
    5: {'male': 0.4986, 'female': 0.5014},  # composer
    6: {'male': 0.4974, 'female': 0.5026},  # dentist
    7: {'male': 0.4980, 'female': 0.5020},  # dietitian
    8: {'male': 0.5187, 'female': 0.4813},  # dj
    9: {'male': 0.4968, 'female': 0.5032},  # filmmaker
    10: {'male': 0.5000, 'female': 0.5000},  # interior_designer
    11: {'male': 0.5012, 'female': 0.4988},  # journalist
    12: {'male': 0.5046, 'female': 0.4954},  # model
    13: {'male': 0.4963, 'female': 0.5037},  # nurse
    14: {'male': 0.4967, 'female': 0.5033},  # painter
    15: {'male': 0.5023, 'female': 0.4977},  # paralegal
    16: {'male': 0.5087, 'female': 0.4913},  # pastor
    17: {'male': 0.5071, 'female': 0.4929},  # personal_trainer
    18: {'male': 0.4968, 'female': 0.5032},  # photographer
    19: {'male': 0.5000, 'female': 0.5000},  # physician
    20: {'male': 0.4922, 'female': 0.5078},  # poet
    21: {'male': 0.4994, 'female': 0.5006},  # professor
    22: {'male': 0.5050, 'female': 0.4950},  # psychologist
    23: {'male': 0.5014, 'female': 0.4986},  # rapper
    24: {'male': 0.4929, 'female': 0.5071},  # software_engineer
    25: {'male': 0.4981, 'female': 0.5019},  # surgeon
    26: {'male': 0.4991, 'female': 0.5009},  # teacher
    27: {'male': 0.4918, 'female': 0.5082}   # yoga_teacher
}
tcav_dnn_concepts = {
    0: {'male': 0.5060, 'female': 0.4940},  # accountant
    1: {'male': 0.4946, 'female': 0.5054},  # architect
    2: {'male': 0.5023, 'female': 0.4977},  # attorney
    3: {'male': 0.5106, 'female': 0.4894},  # chiropractor
    4: {'male': 0.4944, 'female': 0.5056},  # comedian
    5: {'male': 0.4932, 'female': 0.5068},  # composer
    6: {'male': 0.5034, 'female': 0.4966},  # dentist
    7: {'male': 0.5102, 'female': 0.4898},  # dietitian
    8: {'male': 0.4828, 'female': 0.5172},  # dj
    9: {'male': 0.5020, 'female': 0.4980},  # filmmaker
    10: {'male': 0.4891, 'female': 0.5109},  # interior_designer
    11: {'male': 0.5026, 'female': 0.4974},  # journalist
    12: {'male': 0.4912, 'female': 0.5088},  # model
    13: {'male': 0.5042, 'female': 0.4958},  # nurse
    14: {'male': 0.5000, 'female': 0.5000},  # painter
    15: {'male': 0.4988, 'female': 0.5012},  # paralegal
    16: {'male': 0.4798, 'female': 0.5202},  # pastor
    17: {'male': 0.5028, 'female': 0.4972},  # personal_trainer
    18: {'male': 0.4954, 'female': 0.5046},  # photographer
    19: {'male': 0.4995, 'female': 0.5005},  # physician
    20: {'male': 0.4994, 'female': 0.5006},  # poet
    21: {'male': 0.4999, 'female': 0.5001},  # professor
    22: {'male': 0.5033, 'female': 0.4967},  # psychologist
    23: {'male': 0.5000, 'female': 0.5000},  # rapper
    24: {'male': 0.4977, 'female': 0.5023},  # software_engineer
    25: {'male': 0.4963, 'female': 0.5037},  # surgeon
    26: {'male': 0.5012, 'female': 0.4988},  # teacher
    27: {'male': 0.4927, 'female': 0.5073}   # yoga_teacher
}
tcav_scores_dnn_sentencesconcepts = {
    0:{'male': 0.5029, 'female': 0.4971},
    1:{'male': 0.5042, 'female': 0.4958},
    2:{'male': 0.4984, 'female': 0.5016},
    3:{'male': 0.4994, 'female': 0.5006},
    4:{'male': 0.5063, 'female': 0.4937},
    5:{'male': 0.4990, 'female': 0.5010},
    6:{'male': 0.4977, 'female': 0.5023},
    7:{'male': 0.4973, 'female': 0.5027},
    8:{'male': 0.5076, 'female': 0.4924},
    9:{'male': 0.4960, 'female': 0.5040},
    10:{'male': 0.5036, 'female': 0.4964},
    11:{'male': 0.4992, 'female': 0.5008},
    12:{'male': 0.5017, 'female': 0.4983},
    13:{'male': 0.5016, 'female': 0.4984},
    14:{'male': 0.5032, 'female': 0.4968},
    15:{'male': 0.4917, 'female': 0.5083},
    16:{'male': 0.5064, 'female': 0.4936},
    17:{'male': 0.5038, 'female': 0.4962},
    18:{'male': 0.5020, 'female': 0.4980},
    19:{'male': 0.5004, 'female': 0.4996},
    20:{'male': 0.4964, 'female': 0.5036},
    21:{'male': 0.5006, 'female': 0.4994},
    22:{'male': 0.4996, 'female': 0.5004},
    23:{'male': 0.5017, 'female': 0.4983},
    24:{'male': 0.4976, 'female': 0.5024},
    25:{'male': 0.5038, 'female': 0.4962},
    26:{'male': 0.4995, 'female': 0.5005},
    27:{'male': 0.4935, 'female': 0.5065},
}
tcav_scores_paragraph = {
    0: {'Male': 0.5039, 'Female': 0.4961},
    1: {'Male': 0.5018, 'Female': 0.4982},
    2: {'Male': 0.4998, 'Female': 0.5002},
    3: {'Male': 0.4969, 'Female': 0.5031},
    4: {'Male': 0.4958, 'Female': 0.5042},
    5: {'Male': 0.5048, 'Female': 0.4952},
    6: {'Male': 0.5006, 'Female': 0.4994},
    7: {'Male': 0.4986, 'Female': 0.5014},
    8: {'Male': 0.5000, 'Female': 0.5000},
    9: {'Male': 0.4982, 'Female': 0.5018},
    10: {'Male': 0.5026, 'Female': 0.4974},
    11: {'Male': 0.5000, 'Female': 0.5000},
    12: {'Male': 0.4991, 'Female': 0.5009},
    13: {'Male': 0.5014, 'Female': 0.4986},
    14: {'Male': 0.5011, 'Female': 0.4989},
    15: {'Male': 0.4930, 'Female': 0.5070},
    16: {'Male': 0.4932, 'Female': 0.5068},
    17: {'Male': 0.4974, 'Female': 0.5026},
    18: {'Male': 0.5014, 'Female': 0.4986},
    19: {'Male': 0.4984, 'Female': 0.5016},
    20: {'Male': 0.4985, 'Female': 0.5015},
    21: {'Male': 0.4999, 'Female': 0.5001},
    22: {'Male': 0.4986, 'Female': 0.5014},
    23: {'Male': 0.5061, 'Female': 0.4939},
    24: {'Male': 0.5008, 'Female': 0.4992},
    25: {'Male': 0.5016, 'Female': 0.4984},
    26: {'Male': 0.5030, 'Female': 0.4970},
    27: {'Male': 0.5036, 'Female': 0.4964},
}
# Initialize the dictionary to store TCAV scores
tcav_scores_retrain= {
    0: {'Male': 0.5029, 'Female': 0.4971},
    1: {'Male': 0.4883, 'Female': 0.5117},
    2: {'Male': 0.4991, 'Female': 0.5009},
    3: {'Male': 0.4963, 'Female': 0.5037},
    4: {'Male': 0.4993, 'Female': 0.5007},
    5: {'Male': 0.5015, 'Female': 0.4985},
    6: {'Male': 0.4955, 'Female': 0.5045},
    7: {'Male': 0.5072, 'Female': 0.4928},
    8: {'Male': 0.4945, 'Female': 0.5055},
    9: {'Male': 0.5014, 'Female': 0.4986},
    10: {'Male': 0.4973, 'Female': 0.5027},
    11: {'Male': 0.5006, 'Female': 0.4994},
    12: {'Male': 0.5074, 'Female': 0.4926},
    13: {'Male': 0.4999, 'Female': 0.5001},
    14: {'Male': 0.4961, 'Female': 0.5039},
    15: {'Male': 0.4892, 'Female': 0.5108},
    16: {'Male': 0.5038, 'Female': 0.4962},
    17: {'Male': 0.4886, 'Female': 0.5114},
    18: {'Male': 0.4995, 'Female': 0.5005},
    19: {'Male': 0.4988, 'Female': 0.5012},
    20: {'Male': 0.4933, 'Female': 0.5067},
    21: {'Male': 0.4970, 'Female': 0.5030},
    22: {'Male': 0.4990, 'Female': 0.5010},
    23: {'Male': 0.4825, 'Female': 0.5175},
    24: {'Male': 0.5035, 'Female': 0.4965},
    25: {'Male': 0.5003, 'Female': 0.4997},
    26: {'Male': 0.4989, 'Female': 0.5011},
    27: {'Male': 0.5012, 'Female': 0.4988},
}
# Process the TCAV scores
for occupation_id, scores in tcav_scores_final.items():
    gender_gap = scores['female'] - scores['male']
    profession_name = profession_mapping[occupation_id]
    
    results.append({
        'occupation': profession_name,
        'gender_gap': gender_gap,
        'female_percentage': scores['female']/100
    })

# Create DataFrame
results_df = pd.DataFrame(results)

# # Create the plot
# plt.figure(figsize=(15, 10))
# plt.scatter(results_df['female_percentage'], results_df['gender_gap'], marker='o')

# # Add labels for each point
# for _, row in results_df.iterrows():
#     plt.annotate(row['occupation'], 
#                 (row['female_percentage'], row['gender_gap']),
#                 xytext=(5, 5), textcoords='offset points', 
#                 fontsize=8, alpha=0.7)

# # Add reference lines
# plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
# plt.axvline(x=0.005, color='r', linestyle='--', alpha=0.5)

# # Customize the plot
# plt.xlabel('Female TCAV Score (percentage)')
# plt.ylabel('Gender Gap (Female TCAV - Male TCAV)')
# plt.title('Gender Gap vs Female TCAV Score by Profession')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()

# plt.show()

# # Print the results sorted by gender gap
# print("\nResults sorted by gender gap:")
# sorted_results = results_df.sort_values('gender_gap', ascending=False)
# print(sorted_results.to_string(index=False))


# Sort the results by absolute gender gap
results_df['abs_gender_gap'] = abs(results_df['gender_gap'])
sorted_results = results_df.sort_values('abs_gender_gap', ascending=True)

# Create the bar chart
plt.figure(figsize=(15, 10))

# Create bars and color them based on whether they favor male or female
colors = ['red' if x < 0 else 'blue' for x in sorted_results['gender_gap']]
bars = plt.barh(sorted_results['occupation'], sorted_results['gender_gap'], color=colors, alpha=0.6)

# Customize the plot
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Gender Gap (Female TCAV - Male TCAV)')
plt.ylabel('Profession')
plt.title('Gender Bias in Professions (Sorted by Magnitude)')
plt.grid(True, axis='x', alpha=0.3)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.6, label='Female-leaning bias'),
    Patch(facecolor='red', alpha=0.6, label='Male-leaning bias')
]
plt.legend(handles=legend_elements, loc='lower right')

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()

# Print numerical results
print("\nNumerical results sorted by absolute gender gap:")
print(sorted_results[['occupation', 'gender_gap']].to_string(index=False))