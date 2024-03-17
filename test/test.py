import csv
import json
import os
import textwrap


def parse_interproscan_results(data):
    summary = {}
    if 'results' in data:
        for result in data['results']:
            xrefs = result.get('xref', [])
            if not xrefs:
                continue
            protein_id = xrefs[0].get('id', "Unknown Protein ID")
            # Assuming 'sequence' is available at the 'result' level
            protein_sequence = result.get('sequence', '')

            annotations = []
            matches = result.get('matches', [])
            for match in matches:
                signature = match.get('signature', {})
                signature_description = signature.get('description', 'No description available')
                signature_accession = signature.get('accession', 'N/A')

                entry = signature.get('entry', {}) or {}
                go_terms = [{'id': go.get('id'), 'name': go.get('name'), 'database': go.get('db'),
                             'category': go.get('category')} for go in entry.get('goXRefs', [])]

                pathways = [{'name': pathway.get('name'), 'id': pathway.get('id'),
                             'database': pathway.get('databaseName')} for pathway in entry.get('pathwayXRefs', [])]

                match_details = []
                for location in match.get('locations', []):
                    start = location.get('start') - 1  # Assuming 0-based indexing for Python string slicing
                    end = location.get('end')
                    sequence_match = protein_sequence[start:end]  # Extract matched sequence slice
                    match_details.append({'start': start + 1, 'end': end, 'score': location.get('score', 'N/A'),
                                          'evalue': location.get('evalue', 'N/A'), 'sequence_match': sequence_match})

                annotations.append({'signature_accession': signature_accession, 'description': signature_description,
                                    'GO_terms': go_terms, 'pathways': pathways, 'match_details': match_details})

            summary[protein_id] = annotations
    return summary


def write_annotations_to_text(summary, output_file):
    with open(output_file, 'w') as file:
        for protein_id, annotations in summary.items():
            file.write(f"Protein ID: {protein_id}\n")
            if not annotations:  # If there are no annotations for this protein
                file.write("  No matches found.\n\n")
                continue  # Skip to the next protein

            for annotation in annotations:
                # Ensuring values are not None before writing
                signature_accession = annotation['signature_accession'] or "N/A"
                description = annotation['description'] or "No description available"
                file.write(f"  Signature Accession: {signature_accession}\n")
                file.write(f"  Description: {textwrap.fill(description, width=120, subsequent_indent='    ')}\n")

                if annotation['GO_terms']:
                    go_terms_str = ", ".join(
                        [f"{go.get('name', 'N/A')} ({go.get('id', 'N/A')})" for go in annotation['GO_terms']])
                    file.write(f"  GO Terms: {textwrap.fill(go_terms_str, width=120, subsequent_indent='    ')}\n")

                if annotation['pathways']:
                    pathways_str = ", ".join(
                        [f"{pathway.get('name', 'N/A')} ({pathway.get('id', 'N/A')})" for pathway in
                         annotation['pathways']])
                    file.write(f"  Pathways: {textwrap.fill(pathways_str, width=120, subsequent_indent='    ')}\n")

                if annotation['match_details']:
                    match_details_str = "; ".join([f"Start: {md.get('start', 'N/A')}, End: {md.get('end', 'N/A')}, "
                                                   f"Score: {md.get('score', 'N/A')}, Evalue: {md.get('evalue', 'N/A')}, "
                                                   f"Matched Sequence: {md.get('sequence_match', 'N/A')}"
                                                   for md in annotation['match_details']])
                    file.write(
                        f"  Match Details: {textwrap.fill(match_details_str, width=120, subsequent_indent='    ')}\n")

                file.write("\n")


def write_annotations_to_csv(summary, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Protein ID', 'Signature Accession', 'Description', 'GO Terms', 'Pathways', 'Match Details'])

        for protein_id, annotations in summary.items():
            for annotation in annotations:
                go_terms_str = json.dumps(annotation['GO_terms'])  # Serialize list of dicts to JSON string
                pathways_str = json.dumps(annotation['pathways'])  # Serialize list of dicts to JSON string
                match_details_str = json.dumps(annotation['match_details'])  # Serialize list of dicts to JSON string

                writer.writerow([
                    protein_id,
                    annotation['signature_accession'],
                    annotation['description'],
                    go_terms_str,
                    pathways_str,
                    match_details_str
                ])

if __name__ == '__main__':
    output_file = '../results/interpro results/6mer_test1/6mer_test1.json'
    results_directory = '../results/interpro results/6mer_test1/'
    selected_file = '6mer_test1.fasta'
    try:
        with open(output_file) as json_file:
            data = json.load(json_file)
        summary = parse_interproscan_results(data)
        csv_output_file = os.path.join(results_directory, "{}_summary.txt".format(selected_file.removesuffix('.fasta')))
        write_annotations_to_text(summary, csv_output_file)
        print("Annotations summary saved to {}".format(csv_output_file))
    except Exception as e:
        print("An error occurred while parsing or writing the summary: {}".format(e))