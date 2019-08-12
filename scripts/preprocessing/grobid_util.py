from typing import NamedTuple, List, Dict, Optional, Tuple
import bs4
from bs4 import BeautifulSoup


class GrobidBibliographyEntry(NamedTuple):

    raw_xml: bs4.element.Tag
    title: str
    ref_id: str = None

    def authors(self):
        names = []
        for author in self.raw_xml.find_all("author"):

            first = author.persName.forename
            last = author.persName.surname
            names_dict = {}
            if first is not None:
                names_dict["first"] = first.text
            if last is not None:
                names_dict["last"] = last.text

            names.append(names_dict)
        return names


def parse_bibliography(soup: BeautifulSoup) -> List[GrobidBibliographyEntry]:

    """
    Finds all bibliography entries in a grobid xml.
    """
    bibliography = soup.listBibl
    if bibliography is None:
        return []

    entries = bibliography.find_all("biblStruct")

    structured_entries = []    
    for entry in entries:

        title = entry.title.text
        ref_id = entry.attrs.get("xml:id", None)
        structured_entries.append(GrobidBibliographyEntry(entry, title, ref_id))

    return structured_entries


class Section(NamedTuple):
    paragraphs: List[str]
    mention_spans: List[List[Tuple[int, int]]]
    mention_ids: List[List[Optional[str]]]
    name: str = None

def is_reference_tag(tag: bs4.element.Tag):
    return tag.name == "ref" and tag.attrs.get("type", "") == "bibr"

def extract_references_from_paragraph_text(tag: bs4.element.Tag) -> Section:

    """
    Parameters
    ----------
    tag: `bs4.element.Tag`
        The <div/> element of a xml document produced by Grobid.
    
    Returns
    -------
    A Section object containing the paragraphs contained within the section,
    mention spans within the paragraphs, mention ids (if produced by grobid) linking
    the mention spans to bibliography entries, and the name of the section.

    """

    name = None
    paragraphs = []
    reference_spans = []
    reference_ids = []
    for child in tag.children:
        if child.name == "head":
            name = child.text
            continue
        # TODO(Mark): Other tags can occur here, like formula. Should we include them?
        elif child.name == "p":

            paragraph_text = ""
            paragraph_spans = []
            paragraph_ids = []
            index = 0
            for text_child in child.children:
                if isinstance(text_child, str):

                    paragraph_text += text_child
                    index += len(text_child)
                elif isinstance(text_child, bs4.element.Tag) and is_reference_tag(text_child):
                    reference_text = text_child.text
                    start = index
                    index += len(reference_text)
                    end = index
                    paragraph_text += '<pwc_cite>' + reference_text + '</pwc_cite>'

                    paragraph_spans.append((start, end))
                    paragraph_ids.append(text_child.attrs.get("target", None))
                else:
                    # TODO(Mark): Think about whether we want to include these in the full text.
                    # these are things like figure and formula references.
                    paragraph_text += text_child.text
                    index += len(text_child.text)

            paragraphs.append(paragraph_text)
            reference_spans.append(paragraph_spans)
            reference_ids.append(paragraph_ids)

    return Section(paragraphs, reference_spans, reference_ids, name)
