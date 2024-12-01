import os
import pandas as pd
from transition_amr_parser.parse import AMRParser

# Load the AMR parser model
parser = AMRParser.from_pretrained('AMR3-structbart-L')

# Read the CSV file
df = pd.read_csv('../../amr/out/bold_response_LH.csv')  # Replace with your CSV file path

# Assuming the CSV has a column named 'sentence' containing the sentences
sentences = df['sentence'].tolist()
ids = df['item_id'].tolist()

# Specify the directory to save plots
plot_directory = "plot_directory"  # Replace with your desired directory name
output_file = "out/test_output.txt"
output_data =[]


# Open a file to write the Penman notations
for i, sentence in enumerate(sentences):
        # Tokenize the sentence
    tokens, positions = parser.tokenize(sentence)
        
        # Parse the sentence
    annotations, machines = parser.parse_sentence(tokens)
        
        # Print Penman notation in the terminal (optional)
    print(f"Sentence {i+1}: {sentence}")
    print(annotations)
        
        # Get AMR graph object
    amr = machines.get_amr()
        
    penman_notation = amr.to_penman(jamr=False, isi=True)

    output_data.append({
            'item_id': ids[i],
            'penman_notation': penman_notation
        })
        
        # # Plot the AMR graph
        # fig, ax = plt.subplots(figsize=(10, 5))  # You can adjust the figure size
        # amr.plot()
        
        # # Save the plot to a file
        # plot_filename = os.path.join(plot_directory, f"amr_plot_{i+1}.png")
        # plt.savefig(plot_filename)
        # plt.close()

output_df = pd.DataFrame(output_data)
output_df.to_csv('amr_output.csv', index=False)



print("AMR processing completed. Check 'amr_output.txt' and the generated plot images.")
