import requests

from scirex_utilities import io_util

# TODO: the grobid end point -- change in configuration afterwards if important
PDF_PARSER_HOST = ""


def parse_by_grobid(input_dir, paper_id, output_dir):
    with open(io_util.join(input_dir, str(paper_id) + ".pdf"), "rb") as a_file:
        file_dict = {"input": a_file}
        response = requests.post(PDF_PARSER_HOST, files=file_dict, verify=False)

    if response.status_code == 200:
        io_util.write_text_to_file(io_util.join(output_dir, str(paper_id) + ".tei.xml"), response.text)
        return True

    print(response.content)
    return False
