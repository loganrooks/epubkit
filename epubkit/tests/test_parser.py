import dataclasses
from pathlib import Path
import bs4
import pytest
from bs4 import BeautifulSoup
from epubkit.debug import debug_print_dict, log_error, setup_logging
from epubkit.parser import (
    HTMLCategoryExtractor,
    CategoryPattern, 
    CategoryExtractionError,
    CategoryType,
    MatchCriteria,
    PatternReviewBackend,
    TagInfo,
    ImmutableTagInfo,
    html_to_selections,
    CategoryDict,
    HTML,
    TagMatcher,
    TextBlock,
    ExtractedText,
    extract_text_by_headers
)
from typing import Dict, Tuple, List, Optional

from epubkit.tests.utils import analyze_extraction_coverage


@pytest.fixture
def sample_selections():
    return {
        'headers': [('Chapter 1', (
            ImmutableTagInfo('h1', ('chapter-title',), r'ch\d+', (('data-level', '1'),)),
        ))],
        'body': [('Sample text', (
            ImmutableTagInfo('p', ('body-text',), '', ()),
        ))]
    }

@pytest.fixture
def extractor(sample_selections):
    return HTMLCategoryExtractor(sample_selections)

def test_init(sample_selections):
    extractor = HTMLCategoryExtractor(sample_selections)
    assert extractor.selections == sample_selections
    assert len(extractor.category_patterns) > 0

def test_build_category_patterns(extractor):
    patterns = extractor.category_patterns
    
    # Check headers pattern
    assert 'headers' in patterns
    header_pattern = patterns['headers']
    assert isinstance(header_pattern, CategoryPattern)
    assert header_pattern.category == 'headers'
    assert len(header_pattern.root_matchers) > 0
    
    # Check root matcher structure
    root_matcher = next(iter(header_pattern.root_matchers))
    assert isinstance(root_matcher, TagMatcher)
    assert root_matcher.tag == 'h1'
    assert 'chapter-title' in root_matcher.classes
    
    # Verify pattern is immutable
    with pytest.raises(AttributeError):
        header_pattern.root_matchers.add(TagMatcher('p'))
        
def test_complex_pattern_matching(extractor):
    html = """
        <p class="calibre_6">
            <span class="calibre12">Text</span>
        </p>
    """
    soup = BeautifulSoup(html, 'html.parser')
    pattern = CategoryPattern(
        category='body',
        root_matchers={
            TagMatcher(
                tag='p',
                classes=frozenset(['calibre_6']),
                required_children=frozenset([
                    TagMatcher(
                        tag='span',
                        classes=frozenset(['calibre12'])
                    )
                ])
            )
        }
    )
    
    element = soup.find_all(tuple(pattern.root_matchers)[0].tag)[0]
    assert extractor._matches_pattern(element, pattern)

def test_build_css_selector(extractor):
    tag = 'h1'
    classes = {'chapter-title', 'main'}
    selector = extractor._build_css_selector(tag, classes)
    assert selector.startswith('h1')
    assert '.chapter-title' in selector
    assert '.main' in selector

def test_matches_pattern(extractor):
    html = '<h1 class="chapter-title" id="ch1" data-level="1">Chapter 1</h1>'
    soup = BeautifulSoup(html, 'html.parser')
    
    matcher = TagMatcher(
        tag='h1',
        classes=frozenset(['chapter-title']),
        id_pattern=r'ch\d+',
        attrs=(('data-level', '1'),)
    )
    
    pattern = CategoryPattern(
        category='headers',
        root_matchers={matcher},
        global_excluded=set()
    )
    
    element = soup.find_all(matcher.tag)[0]
    assert extractor._matches_pattern(element, pattern) == True

def test_matches_pattern_with_excluded_tags(extractor):
    html = '<h1 class="chapter-title footnote">Chapter 1</h1>'
    soup = BeautifulSoup(html, 'html.parser')
    
    matcher = TagMatcher(
        tag='h1',
        classes=frozenset(['chapter-title'])
    )
    
    excluded = TagMatcher(
        tag='h1',
        classes=frozenset(['footnote'])
    )
    
    pattern = CategoryPattern(
        category='headers',
        root_matchers={matcher},
        global_excluded={excluded}
    )
    
    assert extractor._matches_pattern(soup, pattern) == False

def test_extract_category_success(extractor):
    html = """
    <div>
        <h1 class="chapter-title" id="ch1" data-level="1">Chapter 1</h1>
        <h2 class="section-title" id="sec1" data-level="1.1">Section 1.1</h2>
        <p class="body-text">This is body text</p>
    </div>
    """
    
    results = extractor.extract_category(html)
    
    assert 'headers' in results
    assert len(results['headers']) == 1
    assert results['headers'][0]['text'] == 'Chapter 1'
    
    assert 'body' in results
    assert len(results['body']) == 1
    assert results['body'][0]['text'] == 'This is body text'

def test_extract_category_invalid_html():
    extractor = HTMLCategoryExtractor({'headers': set()})
    
    with pytest.raises(CategoryExtractionError):
        extractor.extract_category("%$%#^#$%$#")

def test_extract_category_empty_html(extractor):
    results = extractor.extract_category("")
    assert all(len(matches) == 0 for matches in results.values())

def test_matches_pattern_with_file_position(extractor):
    html = '<p class="body-text" id="filepos1234">Text</p>'
    soup = BeautifulSoup(html, 'html.parser')
    
    matcher = TagMatcher(
        tag='p',
        classes=frozenset(['body-text']),
        id_pattern=r'filepos\d+'
    )
    
    pattern = CategoryPattern(
        category='body',
        root_matchers={matcher},
        file_position_pattern=r'filepos\d+'
    )
    element = soup.find_all(matcher.tag)[0]
    assert extractor._matches_pattern(element, pattern) == True

def test_extract_category_multiple_matches(extractor):
    html = """
    <div>
        <h1 class="chapter-title" id="ch1" data-level="1">Chapter 1</h1>
        <h1 class="chapter-title" id="ch2" data-level="1">Chapter 2</h1>
    </div>
    """
    
    results = extractor.extract_category(html)
    assert 'headers' in results
    assert len(results['headers']) == 2


class TestHeideggerText:
    selections: CategoryDict[Tuple[str, Tuple[ImmutableTagInfo, ...]]]
    categorized_html: CategoryDict[HTML]
    extractor: HTMLCategoryExtractor

    @pytest.fixture(autouse=True)
    def sample_selections(self) -> Dict[CategoryType, List[Tuple[str, Tuple[ImmutableTagInfo, ...]]]]:
        self.categorized_html = {
            'headers': ["""<p id="filepos41583" class="calibre_8">
                        <span class="calibre11"><span class="bold">
                            I: THE NECESSITY, STRUCTURE, AND PRIORITY OF THE QUESTION OF BEING
                        </span></span>
                    </p>"""],
            'subheaders': ["""<p id="filepos41760" class="calibre_10">
            <span class="calibre11"><span class="bold"><span class="italic">
                <a><span>1. The Necessity for Explicitly Restating the Question of Being</span></a>
            </span></span></span>
        </p>"""],
            'body': ["""
                    <p class="calibre_6">
                        <span class="calibre6">
                            <span><span class="calibre10">THIS </span></span>
                        <span>
                            <span class="calibre10">question has today been forgotten. Even though in our time we deem it progressive to give our approval to ‘metaphysics’ again, it is held that we have been exempted from the exertions of a newly rekindled <span class="italic">γιγαντομαχία περὶ τῆς οὐσίας</span><span>. </span>Yet the question we are touching upon is not just <span> </span><span>a n y</span> question. It is one which provided a stimulus for the researches of Plato and Aristotle, only to subside from then on <span class="italic">as a theme for actual investigation</span><span>.(1) </span>What these two men achieved was to persist through many alterations and ‘retouchings’ down to the ‘logic’ of Hegel. And what they wrested with the utmost intellectual effort from the phenomena, fragmentary and incipient though it was, has long since become trivialized. </span></span></span></p>
                    """,
                    """<p class="calibre_6">
                        <span class="calibre12">
                            <span>
                                <span class="calibre13">
                                    MORE than thirty years have passed since 
                                    <span class="italic">Being and Time</span>
                                    first appeared...
                                </span>
                            </span>
                        </span>
                    </p>
                    """,
                    """<p class="calibre_6">
                        <span class="calibre12"><span><span class="calibre13">
                            Text before <span class="italic">Being and Time</span><span></span>
                            continues after with more content
                        </span></span></span>
                    </p>
                    """],
            'footnotes': ['<p class="calibre_6"><span class="calibre9"><span>1.<span class="italic"> ‘…als thematische Frage wirklicher Untersuchung’.</span></span>When Heidegger speaks of a question as ‘thematisch’, he thinks of it as one which is taken seriously and studied in a systematic manner. While we shall often translate this adjective by its cognate, ‘thematic’, we may sometimes find it convenient to choose more flexible expressions involving the word ‘theme’. (Heidegger gives a fuller discussion on H. 363.)</span></p>',
                          '<p class="calibre_6"><span class="calibre9">4. ‘…als des möglichen Horizontes eines jeden Seinsverständnisses überhaupt…’ Throughout this work the word ‘horizon’ is used with a connotation somewhat different from that to which the English-speaking reader is likely to be accustomed. We tend to think of a horizon as something which we may widen or extend or go beyond; Heidegger, however, seems to think of it rather as something which we can neither widen nor go beyond, but which provides the limits for certain intellectual activities performed ‘within’ it.</span></p>',
                          '<p class="calibre_6"><span class="calibre9">1. While we shall ordinarily reserve the word ‘falling’ for ‘Verfallen’ (see our note 2, H. 21 above), in this sentence it represents first ‘Verfallen’ and then ‘Fallen’, the usual German word for ‘falling’. ‘Fallen’ and ‘Verfallen’ are by no means strictly synonymous; the latter generally has the further connotation of ‘decay’ or ‘deterioration’, though Heidegger will take pains to point out that in his own usage it ‘does not express any negative evaluation’. See Section 38 below. </span></p>'],
            'toc': []
        }
        self.selections = html_to_selections(self.categorized_html)
        self.extractor = HTMLCategoryExtractor(self.selections)

    def test_nested_spans_with_classes(self):
        html = """
        <p class="calibre_6">
            <span class="calibre12">
                <span>
                    <span class="calibre13">
                        MORE than thirty years have passed since 
                        <span class="italic">Being and Time</span>
                        first appeared...
                    </span>
                </span>
            </span>
        </p>
        """
        
        inner_span = TagMatcher(
            tag='span',
            classes=frozenset(['calibre13'])
        )
        
        middle_span = TagMatcher(
            tag='span',
            required_children=frozenset([inner_span])
        )
        
        outer_span = TagMatcher(
            tag='span',
            classes=frozenset(['calibre12']),
            required_children=frozenset([middle_span])
        )
        
        root_matcher = TagMatcher(
            tag='p',
            classes=frozenset(['calibre_6']),
            required_children=frozenset([outer_span])
        )
        
        pattern = CategoryPattern(
            category='body',
            root_matchers={root_matcher}
        )
        
        soup = BeautifulSoup(html, 'html.parser')
        root_element = soup.find_all(root_matcher.tag)[0]
        assert self.extractor._matches_pattern(root_element, pattern) == True

    def test_extract_nested_spans_content(self):
        html = """
        <p class="calibre_6">
            <span class="calibre12"><span><span class="calibre13">
                Test content with <span class="italic">italicized</span> text
            </span></span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 1
        assert "Test content with italicized text" in results['body'][0]['text'].strip()

    def test_multiple_nested_paragraphs(self):
        html = """
        <div>
            <p class="calibre_6">
                <span class="calibre12"><span><span class="calibre13">First paragraph</span></span></span>
            </p>
            <p class="calibre_6">
                <span class="calibre12"><span><span class="calibre13">Second paragraph</span></span></span>
            </p>
        </div>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 2
        assert "First paragraph" in results['body'][0]['text']
        assert "Second paragraph" in results['body'][1]['text']

    def test_nested_spans_with_attributes(self):
        html = """
        <p class="calibre_6">
            <span class="calibre12" id="span1">
                <span data-custom="value">
                    <span class="calibre13">Content</span>
                </span>
            </span>
        </p>
        """
        
        pattern = CategoryPattern(
            category='body',
            root_matchers={TagMatcher(
                tag='p',
                classes=frozenset(['calibre_6']),
                required_children=frozenset([TagMatcher(
                    tag='span',
                    classes=frozenset(['calibre12'])
                )])
            )}
        )
        
        soup = BeautifulSoup(html, 'html.parser')
        element = soup.find_all('p')[0]
        assert self.extractor._matches_pattern(element, pattern) == True

    def test_mixed_content_spans(self):
        html = """
        <p class="calibre_6">
            <span class="calibre12"><span><span class="calibre13">
                Text before <span class="italic">Being and Time</span><span></span>
                continues after with more content
            </span></span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 1
        text = results['body'][0]['text'].strip()
        assert "Text before" in text
        assert "Being and Time" in text
        assert "continues after" in text

    def test_empty_spans(self):
        html = """
        <p class="calibre_6">
            <span class="calibre12">
                <span></span>
                <span class="calibre13">Content</span>
                <span></span>
            </span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 1
        assert results['body'][0]['text'].strip() == "Content"

    def test_heidegger_style_structure(self):
        html = """
        <p id="filepos41583" class="calibre_8">
            <span class="calibre11"><span class="bold">
                I: THE NECESSITY, STRUCTURE, AND PRIORITY OF THE QUESTION OF BEING
            </span></span>
        </p>
        <div class="calibre_9"> </div>
        <p id="filepos41760" class="calibre_10">
            <span class="calibre11"><span class="bold"><span class="italic">
                <a><span>1. The Necessity for Explicitly Restating the Question of Being</span></a>
            </span></span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'headers' in results
        assert 'subheaders' in results
        assert len(results['headers']) == 1
        assert "NECESSITY, STRUCTURE, AND PRIORITY" in results['headers'][0]['text']
        assert len(results['subheaders']) == 1
        assert "Explicitly Restating" in results['subheaders'][0]['text']

    def test_heidegger_single_block(self):
        html = """
         <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">THIS question has today been forgotten. 
                Even though in our time we deem it progressive(1)</span>
            </span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 1
        assert "THIS question has today been forgotten." in results['body'][0]['text']
        assert "Even though in our time we deem it progressive(1)" in results['body'][0]['text']

    def test_heidegger_body_with_footnotes(self):
        html = """
        <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">THIS question has today been forgotten. 
                Even though in our time we deem it progressive(1)</span>
            </span></span>
        </p>
        <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">Not only that. On the basis of the Greeks' 
                initial contributions towards an Interpretation of Being(2)</span>
            </span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 2
        assert "THIS question" in results['body'][0]['text']
        assert "Not only that" in results['body'][1]['text']

    def test_heidegger_italic_formatting(self):
        html = """
        <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">It is held that 
                <span class="italic">γιγαντομαχία περὶ τῆς οὐσίας</span>
                Yet the question we are touching upon</span>
            </span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 1
        assert "γιγαντομαχία περὶ τῆς οὐσίας" in results['body'][0]['text']

    def test_heidegger_page_reference(self):
        html = """
        <p class="calibre_1">
            <span class="calibre11">
                <span class="italic"><span>H. 2</span></span>
            </span>
        </p>
        """
        
        pattern = CategoryPattern(
            category='headers',
            root_matchers={TagMatcher(
                tag='p',
                classes=frozenset(['calibre_1']),
                required_children=frozenset([TagMatcher(
                    tag='span',
                    classes=frozenset(['italic']),
                    match_criteria = MatchCriteria(position_invariant=True)
                )])
            )}
        )
        
        soup = BeautifulSoup(html, 'html.parser')
        element = soup.find_all('p')[0]
        assert self.extractor._matches_pattern(element, pattern) == True

    def test_heidegger_complete_structure(self):
        html = """
        <p id="filepos41583" class="calibre_8">
            <span class="calibre11"><span class="bold">Main Title</span></span>
        </p>
        <div class="calibre_9"> </div>
        <p id="filepos41760" class="calibre_10">
            <span class="calibre11"><span class="bold"><span class="italic">
                <a><span>Subtitle Here</span></a>
            </span></span></span>
        </p>
        <p class="calibre_1">
            <span class="calibre11"><span class="italic"><span>H. 2</span></span></span>
        </p>
        <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">Body text here with 
                <span class="italic">special formatting</span> 
                and continuing text(1)</span>
            </span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert all(category in results for category in ['headers', 'subheaders', 'body'])
        assert len(results['headers']) == 1
        assert len(results['subheaders']) == 1
        assert len(results['body']) == 1
        assert "Main Title" in results['headers'][0]['text']
        assert "Subtitle Here" in results['subheaders'][0]['text']
        assert "Body text here" in results['body'][0]['text']

    def test_heidegger_empty_divs(self):
        html = """
        <p id="filepos41583" class="calibre_8">
                        <span class="calibre11"><span class="bold">
                            I: THE NECESSITY, STRUCTURE, AND PRIORITY OF THE QUESTION OF BEING
                        </span></span>
                    </p> 
        <div class="calibre_9"> </div>
        <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">Body text here with 
                <span class="italic">special formatting</span> 
                and continuing text(1)</span>
                </span></span>        
            <div class="calibre_9"> </div>
        """
        
        results = self.extractor.extract_category(html)
        assert 'headers' in results
        assert 'body' in results
        # Empty divs should not affect extraction
        assert len(results['headers']) == 1
        assert len(results['body']) == 1

    def test_heidegger_nested_formatting(self):
        html = """
        <p class="calibre_6">
            <span class="calibre6"><span>
                <span class="calibre10">Text with 
                <span class="italic">italics</span> and 
                <span class="bold">bold</span> and 
                <span class="italic"><span class="bold">both</span></span>
                formatting</span>
            </span></span>
        </p>
        """
        
        results = self.extractor.extract_category(html)
        assert 'body' in results
        assert len(results['body']) == 1
        text = results['body'][0]['text']
        assert all(phrase in text for phrase in ["Text with", "italics", "bold", "both", "formatting"])


class TestTextExtraction:
    @pytest.fixture
    def sample_extracted_text(self) -> ExtractedText:
        return ExtractedText(
            headers=[
                TextBlock(
                    text="Chapter 1: Introduction",
                    category="headers",
                    header_path=["Chapter 1: Introduction"]
                ),
                TextBlock(
                    text="Chapter 2: Main Concepts",
                    category="headers",
                    header_path=["Chapter 2: Main Concepts"]
                )
            ],
            subheaders=[
                TextBlock(
                    text="1.1 Background",
                    category="subheaders",
                    header_path=["Chapter 1: Introduction", "1.1 Background"]
                ),
                TextBlock(
                    text="2.1 Core Ideas",
                    category="subheaders",
                    header_path=["Chapter 2: Main Concepts", "2.1 Core Ideas"]
                )
            ],
            body=[
                TextBlock(
                    text="This is the introduction text.",
                    category="body",
                    header_path=["Chapter 1: Introduction"],
                    footnotes=["First footnote"]
                ),
                TextBlock(
                    text="Here are the main concepts.",
                    category="body",
                    header_path=["Chapter 2: Main Concepts"],
                    footnotes=["Second footnote"]
                )
            ],
            footnotes=[
                TextBlock(
                    text="First footnote",
                    category="footnotes"
                ),
                TextBlock(
                    text="Second footnote",
                    category="footnotes"
                )
            ],
            toc=[]
        )

    def test_save_extracted_text(self, sample_extracted_text, tmp_path):
        """Test saving extracted text to files"""
        output_dir = tmp_path / "test_output"
        sample_extracted_text.save_to_file(output_dir)
        
        # Check that files were created
        assert (output_dir / "headers.txt").exists()
        assert (output_dir / "body.txt").exists()
        assert (output_dir / "footnotes.txt").exists()
        
        # Verify content format
        with open(output_dir / "headers.txt", 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Chapter 1: Introduction" in content
            assert "# Chapter 2: Main Concepts" in content

    def test_extract_text_by_headers_end_footnotes(self, sample_extracted_text, monkeypatch):
        """Test extracting text organized by headers with footnotes at end"""
        # Mock extract_categorized_text to return our sample data
        monkeypatch.setattr(
            "epubkit.parser.extract_categorized_text",
            lambda *args: sample_extracted_text
        )
        
        organized = extract_text_by_headers("dummy.epub", footnote_mode='end')
        
        assert "Chapter 1: Introduction" in organized
        assert "Chapter 2: Main Concepts" in organized
        
        ch1_content = organized["Chapter 1: Introduction"]
        assert "This is the introduction text." in ch1_content
        assert "Footnotes:" in ch1_content
        assert "[1] First footnote" in ch1_content

    def test_extract_text_by_headers_inline_footnotes(self, sample_extracted_text, monkeypatch):
        """Test extracting text with inline footnotes"""
        monkeypatch.setattr(
            "epubkit.parser.extract_categorized_text",
            lambda *args: sample_extracted_text
        )
        
        organized = extract_text_by_headers("dummy.epub", footnote_mode='inline')
        
        ch2_content = organized["Chapter 2: Main Concepts"]
        assert "Here are the main concepts." in ch2_content
        assert "[1] Second footnote" in ch2_content
        assert "Footnotes:" not in ch2_content  # Should not have footnotes section

    def test_extract_text_by_headers_ignore_footnotes(self, sample_extracted_text, monkeypatch):
        """Test extracting text while ignoring footnotes"""
        monkeypatch.setattr(
            "epubkit.parser.extract_categorized_text",
            lambda *args: sample_extracted_text
        )
        
        organized = extract_text_by_headers("dummy.epub", footnote_mode='ignore')
        
        for content in organized.values():
            assert "footnote" not in content.lower()
            assert "[1]" not in content

    def test_extract_text_by_headers_with_subheaders(self, sample_extracted_text, monkeypatch):
        """Test that subheaders are properly nested under main headers"""
        monkeypatch.setattr(
            "epubkit.parser.extract_categorized_text",
            lambda *args: sample_extracted_text
        )
        
        organized = extract_text_by_headers("dummy.epub")
        
        ch1_content = organized["Chapter 1: Introduction"]
        assert "## 1.1 Background" in ch1_content
        
        ch2_content = organized["Chapter 2: Main Concepts"]
        assert "## 2.1 Core Ideas" in ch2_content

    def test_saving_organized_text(self, sample_extracted_text, tmp_path, monkeypatch):
        """Test saving organized text to file"""
        monkeypatch.setattr(
            "epubkit.parser.extract_categorized_text",
            lambda *args: sample_extracted_text
        )
        
        output_dir = tmp_path / "organized"
        organized = extract_text_by_headers(
            "dummy.epub",
            output_path=output_dir
        )
        
        # Check that file was created
        output_file = output_dir / "organized_text.txt"
        assert output_file.exists()
        
        # Verify content structure
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Chapter 1: Introduction" in content
            assert "## 1.1 Background" in content
            assert "=" * 80 in content  # Section separator
            assert "Footnotes:" in content

    def test_header_path_handling(self, sample_extracted_text, monkeypatch):
        """Test proper handling of nested header paths"""
        monkeypatch.setattr(
            "epubkit.parser.extract_categorized_text",
            lambda *args: sample_extracted_text
        )
        
        organized = extract_text_by_headers("dummy.epub")
        
        # Headers should be used as keys
        assert "Chapter 1: Introduction" in organized
        assert "Chapter 2: Main Concepts" in organized
        
        # Content should be under correct headers
        assert "This is the introduction text." in organized["Chapter 1: Introduction"]
        assert "Here are the main concepts." in organized["Chapter 2: Main Concepts"]

class TestBeingAndTimeContent:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.html_map = {
            'THIS question has today been forgotten...': {
                'html': '''<p class="calibre_6">
                    <span class="calibre6"><span><span class="calibre10">THIS question has today been forgotten...</span></span></span>
                </p>''',
                'tag_hierarchy': [
                    TagInfo('p', ['calibre_6'], '', []),
                    TagInfo('span', ['calibre6'], '', []),
                    TagInfo('span', [], '', []),
                    TagInfo('span', ['calibre10'], '', [])
                ]
            },
            '1. When Heidegger speaks of a question...': {
                'html': '''<p class="calibre_6"><span class="calibre9">
                    <span>1.<span class="italic"> '…als thematische Frage wirklicher Untersuchung'.</span></span>
                    When Heidegger speaks of a question...</span></p>''',
                'tag_hierarchy': [
                    TagInfo('p', ['calibre_6'], '', []),
                    TagInfo('span', ['calibre9'], '', []),
                    TagInfo('span', [], '', []),
                    TagInfo('span', ['italic'], '', [])
                ]
            }
        }

        self.selections = {
            'body': [('THIS question has today been forgotten...', 
                     tuple(ImmutableTagInfo.from_tag_info(t) 
                           for t in self.html_map['THIS question has today been forgotten...']['tag_hierarchy']))],
            'footnotes': [('1. When Heidegger speaks of a question...',
                          tuple(ImmutableTagInfo.from_tag_info(t)
                                for t in self.html_map['1. When Heidegger speaks of a question...']['tag_hierarchy']))]
        }
        
        self.extractor = HTMLCategoryExtractor(self.selections)

    def test_body_text_extraction(self):
        html = self.html_map['THIS question has today been forgotten...']['html']
        results = self.extractor.extract_category(html)
        
        assert 'body' in results
        assert len(results['body']) == 1
        assert 'THIS question has today been forgotten' in results['body'][0]['text']

    def test_footnote_extraction(self):
        html = self.html_map['1. When Heidegger speaks of a question...']['html']
        results = self.extractor.extract_category(html)
        
        assert 'footnotes' in results
        assert len(results['footnotes']) == 1
        assert '1. When Heidegger speaks of a question' in results['footnotes'][0]['text']

    def test_pattern_matching(self):
        pattern = PatternReviewBackend(self.selections, self.html_map)
        pattern.run_initial_tests()
        
        assert 'body' in pattern.test_results
        assert 'footnotes' in pattern.test_results
        assert len(pattern.test_results['body']['matches']) > 0
        assert len(pattern.test_results['footnotes']['matches']) > 0

def test_pattern_review_backend_initialization():
    """Test PatternReviewBackend initialization and pattern loading"""
    selections = {
        'headers': [('Test Header', (ImmutableTagInfo('h1', ('title',), '', ()),))],
        'body': [('Test Body', (ImmutableTagInfo('p', ('content',), '', ()),))]
    }
    html_map = {
        'Test Header': {'html': '<h1 class="title">Test Header</h1>', 'tag_hierarchy': []},
        'Test Body': {'html': '<p class="content">Test Body</p>', 'tag_hierarchy': []}
    }
    
    backend = PatternReviewBackend(selections, html_map)
    assert backend.selections == selections
    assert backend.html_map == html_map
    assert isinstance(backend.extractor, HTMLCategoryExtractor)
    assert backend.extractor.category_patterns

def test_pattern_review_initial_tests():
    """Test initial pattern matching and categorization"""
    selections = {
        'headers': [('Chapter 1', (ImmutableTagInfo('h1', ('chapter',), '', ()),))],
        'body': [('Content', (ImmutableTagInfo('p', ('text',), '', ()),))]
    }
    html_map = {
        'Chapter 1': {'html': '<h1 class="chapter">Chapter 1</h1>', 'tag_hierarchy': []},
        'Content': {'html': '<p class="text">Content</p>', 'tag_hierarchy': []},
        'New Text': {'html': '<p class="text">New Text</p>', 'tag_hierarchy': []}
    }
    
    backend = PatternReviewBackend(selections, html_map)
    backend.run_initial_tests()
    
    assert 'headers' in backend.test_results
    assert 'body' in backend.test_results
    assert len(backend.test_results['headers']['matches']) > 0
    assert len(backend.test_results['body']['matches']) > 0

def test_pattern_review_test_pattern():
    """Test pattern testing logic"""
    selections = {
        'headers': [('Test Header', (ImmutableTagInfo('h1', ('title',), '', ()),))]
    }
    html_map = {
        'Test Header': {'html': '<h1 class="title">Test Header</h1>', 'tag_hierarchy': []},
        'Another Header': {'html': '<h1 class="title">Another Header</h1>', 'tag_hierarchy': []}
    }
    
    backend = PatternReviewBackend(selections, html_map)
    test_pattern = r'<h1 class="title">[^<]*</h1>'
    
    matches, false_positives = backend.test_pattern_logic('headers', test_pattern)
    assert len(matches) == 2  # Should match both headers
    assert not false_positives  # No false positives expected

def test_pattern_review_category_conflicts():
    """Test detection of category conflicts"""
    selections = {
        'headers': [('Conflict Text', (ImmutableTagInfo('h1', ('dual',), '', ()),))],
        'body': [('Conflict Text', (ImmutableTagInfo('p', ('dual',), '', ()),))]
    }
    html_map = {
        'Conflict Text': {'html': '<h1 class="dual">Conflict Text</h1>', 'tag_hierarchy': []}
    }
    
    backend = PatternReviewBackend(selections, html_map)
    backend.run_initial_tests()
    
    assert 'conflicts' in backend.test_results['headers']
    assert 'conflicts' in backend.test_results['body']



from bs4 import BeautifulSoup
import pytest
from epubkit.parser import extract_supervised_patterns, extract_unsupervised_patterns
from epubkit.utils import load_epub_html

def create_test_tag(html_str):
    """Helper to convert HTML string to BS4 tag"""
    return BeautifulSoup(html_str, 'html.parser').find()

def test_supervised_pattern_extraction():
    # Test data structure
    labeled_selections = {
        'headers': [create_test_tag(p) for p in ["""
            <p id="filepos41583" class="calibre_8">
                <span class="calibre11"><span class="bold">
                    I: THE NECESSITY, STRUCTURE, AND PRIORITY OF THE QUESTION OF BEING
                </span></span>
            </p>
        """, """<p class="calibre_8">
                <span class="calibre6">
                    <a></a>
                        <a><span><span><span class="calibre15">
                            <span class="bold"><span>
                                <span class="calibre10">II: THE TWOFOLD TASK IN WORKING OUT THE QUESTION OF BEING. METHOD AND DESIGN OF OUR INVESTIGATION
                                    </span></span></span></span></span></span></a><span><span class="calibre15"><span class="bold"><span><span class="calibre10"></span></span></span></span></span></span></p>"""]
        ],
        'subheaders': [create_test_tag(p) for p in ["""
            <p id="filepos41760" class="calibre_10">
                <span class="calibre11"><span class="bold"><span class="italic">
                    <a><span>1. The Necessity for Explicitly Restating the Question of Being</span></a>
                </span></span></span>
            </p>
        """, """<p id="filepos108674" class="calibre_10">
                    <span class="calibre11"><span class="bold">
                        <span class="italic">
                            <a><span>5. The Ontological Analytic of Dasein as Laying Bare the Horizon for an Interpretation of the Meaning of Being in General
                        </span></a><span>
                    </span><span></span>
                </span></span></span></p>"""]],
        'body': [create_test_tag(p) for p in [
            """<p class="calibre_6"><span class="calibre6"><span><span class="calibre10">THIS </span></span><span><span class="calibre10">question has today been forgotten. Even though in our time we deem it progressive to give our approval to ‘metaphysics’ again, it is held that we have been exempted from the exertions of a newly rekindled <span class="italic">γιγαντομαχία περὶ τῆς οὐσίας</span><span>. </span>Yet the question we are touching upon is not just <span> </span><span>a n y</span> question. It is one which provided a stimulus for the researches of Plato and Aristotle, only to subside from then on <span class="italic">as a theme for actual investigation</span><span>.(1) </span>What these two men achieved was to persist through many alterations and ‘retouchings’ down to the ‘logic’ of Hegel. And what they wrested with the utmost intellectual effort from the phenomena, fragmentary and incipient though it was, has long since become trivialized. </span></span></span></p>""",
            """<p class="calibre_6"><span class="calibre6"><span><span class="calibre10">The question of the meaning of Being must be <span class="italic">formulated</span><span>. </span>If it is a fundamental question, or indeed <span class="italic">the</span><span>
                    </span>fundamental question, it must be made transparent, and in an appropriate way.(1) We must therefore explain briefly what belongs to any question whatsoever, so that from this standpoint the question of Being can be made visible as a <span class="italic">very special</span><span>
                        </span>one with its own distinctive character. </span></span></span></p>""",
            """<p class="calibre_6"><span class="calibre12"><span><span class="calibre13">Inquiry, as a kind of seeking, must be guided beforehand by what is sought. So the meaning of Being must already be available to us in some way. As we have intimated, we always conduct our activities in an understanding of Being. Out of this understanding arise both the explicit question of the meaning of Being and the tendency that leads us towards its conception. We do not <span class="italic">know</span><span>
                    </span>what ‘Being’ means. But even if we ask, ‘What <span class="italic">is</span><span>
                        </span>“Being”?’, we keep within an understanding of the ‘is’, though we are unable to fix conceptually what that ‘is’ signifies. We do not even know the horizon in terms of which that meaning is to be grasped and fixed. <span class="italic">But this vague average understanding of Being is still a Fact. </span></span></span></span></p>""",
            """<p class="calibre_6"><span class="calibre12"><span><span class="calibre13">MORE than thirty years have passed since <span class="italic">Being and Time</span> first appeared...</span></span></span></p>""",
        ]],
        'footnotes': [create_test_tag(f) for f in [
            """<p class="calibre_6"><span class="calibre9">1. First footnote</span></p>""",
            """<p class="calibre_6"><span class="calibre9">4. Second footnote</span></p>"""
        ]]
    }

    result = extract_supervised_patterns(labeled_selections)
    
    # Test results
    assert result['status'] == 'success'
    assert 'patterns' in result
    
    patterns = result['patterns']
    assert len(patterns) == 4  # Should have patterns for all categories
    
    # Test specific patterns
    assert any(p['name'] == 'p' for p in patterns['headers'])
    assert any(p['class'] == ['calibre_8'] for p in patterns['headers'])
    
    assert any(p['name'] == 'p' for p in patterns['body'])
    assert any(p['class'] == ['calibre_6'] for p in patterns['body'])
    
    assert any(p['name'] == 'p' for p in patterns['footnotes'])
    assert any('calibre9' in p.get('class', []) for p in patterns['footnotes'])

def test_unsupervised_pattern_extraction():
    # Load sample HTML from epub
    test_epub_path = str(Path("/home/rookslog/epubkit/epubkit/resources/epubs/Being and Time - Martin Heidegger.epub").resolve())
    html_contents = load_epub_html(test_epub_path, (9, 10))
    
    # Test each HTML document
    for html in html_contents:
        patterns = extract_unsupervised_patterns(html)
        
        # Basic validation
        assert isinstance(patterns, dict)
        assert len(patterns) > 0

        debug_print_dict(patterns)

        analyze_extraction_coverage(html, patterns)
        # Check pattern structure
        for pattern_key, matches in patterns.items():
            assert isinstance(pattern_key, tuple)
            assert all(isinstance(m, bs4.Tag) for m in matches)
            
            # Verify pattern components
            for component in pattern_key:
                name, classes = component
                assert isinstance(name, str)
                assert isinstance(classes, tuple)
                
        # Check common patterns exist
        pattern_names = set(name for pattern in patterns.keys() 
                          for name, _ in pattern)
        assert 'p' in pattern_names  # Should find paragraph patterns
        
def test_empty_input():
    """Test edge cases"""
    assert extract_supervised_patterns({}) == {'status': 'success', 'patterns': {}}
    assert extract_unsupervised_patterns("") == {}


import pytest
from bs4 import BeautifulSoup
from pathlib import Path
from epubkit.parser import TOCExtractor, TOCEntry

@pytest.fixture
def heidegger_toc():
    """Load Heidegger TOC HTML fixture"""
    return """<body class="calibre">
    <h1 id="filepos100" class="calibre_">Contents</h1><ul class="calibre1">
    <li class="calibre2"><a href="index_split_004.html#filepos18830">TRANSLATORS’ PREFACE </a></li>
    <li class="calibre2"><a href="index_split_005.html#filepos30542">AUTHOR’S PREFACE TO THE SEVENTH GERMAN EDITION </a></li>
    <li class="calibre2"><a href="index_split_007.html#filepos40983">INTRODUCTION: EXPOSITION OF THE QUESTION OF THE MEANING OF BEING </a></li>
        <ul class="calibre3"><li class="calibre2"><a href="index_split_008.html#filepos41583">I: THE NECESSITY, STRUCTURE, AND PRIORITY OF THE QUESTION OF BEING </a></li>
            <ul class="calibre4">
                <li class="calibre2"><a href="index_split_008.html#filepos41760">1. The Necessity for Explicitly Restating the Question of Being </a></li>
            </ul>
        </ul>
    </ul></body>"""

@pytest.fixture
def plato_toc():
    """Load Plato TOC HTML fixture"""
    return """
    <body>
<p class="ch"><a id="page_v"/><i>XXXXXXXXXXXXX</i></p>
<p class="centera"><a href="07_Foreword.xhtml"><small>TRANSLATORS' FOREWORD</small></a></p>
<p class="center"><a href="08_Part01.xhtml"><b>PREPARATORY PART</b></a></p>
<p class="center"><a href="08_Part01.xhtml"><i><b>The Essence of Philosophy and the Question of Truth</b></i></a></p>
<p class="toc"><a href="09_Chapter01.xhtml"><b>Chapter One</b>  Preliminary Interpretation of the Essence of Philosophy</a></p>
<p class="toca"><a href="09_Chapter01.xhtml#ch1_1"><i><b>§  1. Futural philosophy; restraint as the basic disposition of the relation to Being</b></i> [<b>Seyn</b>]</a></p>
<p class="toca"><a href="09_Chapter01.xhtml#ch1_2"><i><b>§  2.  Philosophy as the immediately useless, though sovereign, knowledge of the essence of beings</b></i></a></p>
<p class="toca"><a href="09_Chapter01.xhtml#ch1_3"><i><b>§  3.  Questioning the truth of Being, as sovereign knowledge</b></i></a></p>
<p class="toc"><a href="10_Chapter02.xhtml"><b>Chapter Two</b>  The Question of Truth as a Basic Question</a></p>
<p class="toca"><a href="10_Chapter02.xhtml#ch2_1"><i><b>§  4.  Truth as a “problem” of “logic” (correctness of an assertion) distorts every view of the essence of truth</b></i></a></p>
<p class="toca"><a href="10_Chapter02.xhtml#ch2_2"><i><b>§  5.  Discussion of truth by asking the basic question of philosophy, including a historical confrontation with Western philosophy. The need and the necessity of an original questioning</b></i></a></p>
<p class="toc1"><a id="page_vi"/><a href="10_Chapter02.xhtml#ch2_3"><small>RECAPITULATION</small></a></p>
<p class="toc2"><a href="10_Chapter02.xhtml#ch2_4">1)  The question of truth as the most necessary philosophical question in an age that is totally unquestioning</a></p>
<p class="toc2"><a href="10_Chapter02.xhtml#ch2_5">2)  What is worthy of questioning in the determination of truth hitherto (truth as the correctness of an assertion) as compelling us toward the question of truth</a></p>
<p class="toca"><a href="10_Chapter02.xhtml#ch2_6"><i><b>§  6.  The traditional determination of truth as correctness</b></i></a></p>
<p class="toca"><a href="10_Chapter02.xhtml#ch2_7"><i><b>§  7.  The controversy between idealism and realism on the common soil of a conception of truth as the correctness of a representation</b></i></a></p>
<p class="toca"><a href="10_Chapter02.xhtml#ch2_8"><i><b>§  8.  The space of the fourfold-unitary openness. First directive toward what is worthy of questioning in the traditional determination of truth as correctness</b></i></a></p>
<p class="toca"><a href="10_Chapter02.xhtml#ch2_9"><i><b>§  9.  The conception of truth and of the essence of man. The basic question of truth</b></i></a></p>
<p class="toc2"><a href="10_Chapter02.xhtml#ch2_10">a)  The determination of the essence of truth as connected to the determination of the essence of man</a></p>
<p class="toc2"><a href="10_Chapter02.xhtml#ch2_11">b)  The question of the ground of the possibility of all correctness as the basic question of truth</a></p>
<p class="toc1"><a href="10_Chapter02.xhtml#ch2_12"><small>RECAPITULATION</small></a></p>
<p class="toc2"><a href="10_Chapter02.xhtml#ch2_13">1)  The relation between question and answer in the domain of philosophy</a></p>
<p class="toc2"><a href="10_Chapter02.xhtml#ch2_14">2)  The customary determination of truth as correctness of representation, and the fourfold-unitary openness as the question-worthy ground of the possibility of the correctness of representation</a></p>
<p class="toc2"><a id="page_vii"/><a href="10_Chapter02.xhtml#ch2_15">c)  The question of truth as the most questionable of our previous history and the most worthy of questioning of our future history</a></p>"""

@pytest.fixture
def toc_extractor():
    """Create TOCExtractor instance with mock EPUB"""
    epub_path = str(Path("/home/rookslog/epubkit/epubkit/resources/epubs/Being and Time - Martin Heidegger.epub"))
    return TOCExtractor(str(epub_path))

class TestTOCIdentification:
    def test_finds_toc_by_title(self, toc_extractor, heidegger_toc):
        """Test finding TOC page by 'Contents' title"""
        # Mock book items
        toc_extractor.book.get_items_of_type = lambda x: [
            type('MockItem', (), {
                'get_content': lambda: heidegger_toc.encode(),
                'file_name': 'toc.html'
            })
        ]
        
        candidates = toc_extractor.find_toc_candidates()
        assert len(candidates) == 1
        assert candidates[0][0] == 'toc.html'

    def test_finds_toc_by_structure(self, toc_extractor: TOCExtractor, plato_toc):
        """Test finding TOC by hierarchical links"""
        # Mock book items 
        toc_extractor.book.get_items_of_type = lambda x: [
            type('MockItem', (), {
                'get_content': lambda: plato_toc.encode(),
                'file_name': 'toc.html'
            })
        ]
        
        candidates = toc_extractor.find_toc_candidates()
        assert len(candidates) == 1

class TestPatternMatching:
    def test_matches_heidegger_patterns(self, toc_extractor):
        """Test matching Heidegger TOC patterns"""
        tag = BeautifulSoup("""<li class="calibre2">
            <a href="#">Test</a>
        </li>""", 'html.parser').li
        
        pattern = {
            'tag': 'li',
            'classes': ['calibre2']
        }
        
        assert toc_extractor._matches_toc_pattern(tag, pattern)

    def test_matches_plato_patterns(self, toc_extractor):
        """Test matching Plato TOC patterns"""
        tag = BeautifulSoup("""<p class="FM_TocAHead">
            <a href="#">Test</a>
        </p>""", 'html.parser').p
        
        pattern = {
            'tag': 'p',
            'classes': ['FM_TocAHead']
        }
        
        assert toc_extractor._matches_toc_pattern(tag, pattern)

class TestStructureExtraction:
    def test_heidegger_hierarchy(self, toc_extractor: TOCExtractor, heidegger_toc):
        """Test extracting nested hierarchy from Heidegger format"""
        toc_extractor.set_toc_page(heidegger_toc)
        
        patterns = {
            0: {'tag': 'li', 'classes': ['calibre2']},
            1: {'tag': 'li', 'classes': ['calibre2']},
            2: {'tag': 'li', 'classes': ['calibre2']}
        }
        
        entries = toc_extractor.extract_toc_structure(patterns)
        
        assert len(entries) == 3  # Top level entry
        assert len(entries[2].children) == 1  # Second level
        assert len(entries[2].children[0].children) == 1  # Third level

    def test_plato_hierarchy(self, toc_extractor, plato_toc):
        """Test extracting hierarchy from Plato format"""
        toc_extractor.set_toc_page(plato_toc)
        
        patterns = {
            0: {'tag': 'p', 'classes': ['FM_TocPart']},
            1: {'tag': 'p', 'classes': ['FM_TocAHead']},
            2: {'tag': 'p', 'classes': ['FM_TocBHead']}
        }
        
        entries = toc_extractor.extract_toc_structure(patterns)
        
        assert len(entries) == 1
        assert entries[0].title == "Preliminary Considerations"
        assert entries[0].children[0].title == "§1 First Chapter"

class TestTextExtraction:
    def test_extracts_text_blocks(self, toc_extractor):
        """Test extracting text blocks for entries"""
        # Mock book content
        toc_extractor.book.get_items_of_type = lambda x: [
            type('MockItem', (), {
                'get_content': lambda: b"""
                    <p>Valid text block</p>
                    <p>Another block</p>
                    <p class="pagenum">123</p>
                """,
                'file_name': 'content.html'
            })
        ]
        
        entry = TOCEntry(
            title="Test Entry",
            href="content.html",
            level=0
        )
        
        toc_extractor.toc_structure = [entry]
        toc_extractor.extract_text_blocks()
        
        assert len(entry.text_blocks) == 2
        assert "Valid text block" in entry.text_blocks[0]
        assert "Another block" in entry.text_blocks[1]

class TestArtifactFiltering:
    def test_filters_artifacts(self, toc_extractor):
        """Test filtering out common artifacts"""
        artifacts = [
            "123",  # Page number
            "",  # Empty line
            "•",  # Bullet
            "IV",  # Roman numeral
            "\x0c"  # Page break
        ]
        
        for artifact in artifacts:
            assert toc_extractor._is_artifact(artifact)

    def test_keeps_valid_text(self, toc_extractor):
        """Test keeping valid text content"""
        valid_texts = [
            "Regular paragraph text",
            "Chapter 1: Introduction",
            "Section 1.2 Details"
        ]
        
        for text in valid_texts:
            assert not toc_extractor._is_artifact(text)

    def test_custom_artifact_patterns(self, toc_extractor):
        """Test adding custom artifact patterns"""
        toc_extractor._add_artifact_pattern(r"^Chapter \d+$")
        
        assert toc_extractor._is_artifact("Chapter 1")
        assert not toc_extractor._is_artifact("Chapter 1: Content")

class TestParagraphMerging:
    """Test paragraph merging functionality"""
    
    def test_merge_incomplete_sentences(self, toc_extractor):
        """Test merging incomplete sentences"""
        text_blocks = [
            "This is an incomplete",
            "sentence that should merge.",
            "This is complete.",
            "Another incomplete",
            "fragment to merge."
        ]
        html_blocks = [
            "<p>This is an incomplete</p>",
            "<p>sentence that should merge.</p>",
            "<p>This is complete.</p>",
            "<p>Another incomplete</p>",
            "<p>fragment to merge.</p>"
        ]
        
        merged_text, merged_html = toc_extractor._merge_blocks(text_blocks, html_blocks)
        
        assert len(merged_text) == 3
        assert merged_text[0] == "This is an incomplete sentence that should merge."
        assert merged_text[1] == "This is complete."
        assert merged_text[2] == "Another incomplete fragment to merge."
        
        assert len(merged_html) == 3
        assert merged_html[0] == "<p>This is an incomplete</p><p>sentence that should merge.</p>"

    def test_empty_blocks(self, toc_extractor: TOCExtractor):
        """Test handling empty block lists"""
        merged_text, merged_html = toc_extractor._merge_blocks([], [])
        assert len(merged_text) == 0
        assert len(merged_html) == 0

    def test_single_block(self, toc_extractor: TOCExtractor):
        """Test handling single block"""
        text_blocks = ["Complete sentence."]
        html_blocks = ["<p>Complete sentence.</p>"]
        
        merged_text, merged_html = toc_extractor._merge_blocks(text_blocks, html_blocks)
        assert len(merged_text) == 1
        assert merged_text[0] == "Complete sentence."

class TestTextBlockExtraction:
    """Test text block extraction and assignment"""
    
    def test_extract_with_html(self, toc_extractor):
        """Test parallel extraction of text and HTML"""
        # Mock book content
        toc_extractor.book.get_items_of_type = lambda x: [
            type('MockItem', (), {
                'get_content': lambda: b"""
                    <div>
                        <p>First incomplete</p>
                        <p>sentence fragment.</p>
                        <p>Complete sentence here.</p>
                        <p class="header">Section Title</p>
                        <p>More content.</p>
                    </div>
                """,
                'file_name': 'content.html'
            })
        ]
        
        # Create test TOC structure
        entry = TOCEntry(
            title="Section Title",
            href="content.html",
            level=0
        )
        toc_extractor.toc_structure = [entry]
        
        # Extract blocks
        toc_extractor.extract_text_blocks()
        
        # Verify results
        assert len(entry.text_blocks) == 2
        assert entry.text_blocks[0] == "First incomplete sentence fragment."
        assert entry.text_blocks[1] == "Complete sentence here."
        
        assert len(entry.html_blocks) == 2
        assert "<p>First incomplete</p><p>sentence fragment.</p>" in entry.html_blocks[0]

    def test_artifact_filtering(self, toc_extractor):
        """Test filtering of artifacts during extraction"""
        toc_extractor.book.get_items_of_type = lambda x: [
            type('MockItem', (), {
                'get_content': lambda: b"""
                    <div>
                        <p>123</p>
                        <p>Valid content.</p>
                        <p>IV</p>
                        <p>More valid text.</p>
                    </div>
                """,
                'file_name': 'content.html'
            })
        ]
        
        entry = TOCEntry(
            title="Test",
            href="content.html",
            level=0
        )
        toc_extractor.toc_structure = [entry]
        
        toc_extractor.extract_text_blocks()
        
        assert len(entry.text_blocks) == 2
        assert "Valid content." in entry.text_blocks[0]
        assert "More valid text." in entry.text_blocks[1]



import pytest
from ebooklib import epub
from bs4 import BeautifulSoup
from epubkit.parser import TOCEntry, TOCExtractor
from pathlib import Path


def create_test_epub():
    """Create test EPUB with section content"""
    try:
        # Initialize book
        book = epub.EpubBook()
        book.set_identifier('test123')
        book.set_title('Test Book')
        book.set_language('en')

        # Load section content
        example_path = Path(__file__).parent / "examples" / "ebooksection01.xhtml"
        with open(example_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create content pages
        nav = epub.EpubNav(uid='nav')
        nav_content = '''<?xml version="1.0" encoding="utf-8"?>
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
            <head>
                <title>Nav</title>
            </head>
            <body>
                <nav epub:type="toc">
                    <h1>Table of Contents</h1>
                    <ol>
                        <li><a href="chap1.xhtml#filepos242677">I: EXPOSITION</a>
                            <ol>
                                <li><a href="chap1.xhtml#filepos243015">9. The Theme</a></li>
                            </ol>
                        </li>
                    </ol>
                </nav>
            </body>
        </html>'''
        nav.content = nav_content

        # Create main content chapter
        c1 = epub.EpubHtml(
            title='Analysis of Dasein',
            file_name='chap1.xhtml',
            content=content,
            uid='chap1'
        )

        # Add required files
        book.add_item(epub.EpubNcx())
        book.add_item(nav)
        book.add_item(c1)

        # Define Table of Contents
        book.toc = [
            (epub.Section('I: EXPOSITION OF THE TASK OF A PREPARATORY ANALYSIS OF DASEIN'), [
                c1,
                (epub.Link(
                    '9. The Theme of the Analytic of Dasein',
                    'chap1.xhtml#filepos243015',
                    '243015'
                ))
            ])
        ]

        # Basic spine
        book.spine = ['nav', c1]
        book.add_item(epub.EpubNcx())

        # Save EPUB file
        epub_path = Path(__file__).parent / "temp" / "test.epub"
        epub_path.parent.mkdir(exist_ok=True)
        epub.write_epub(str(epub_path), book)
        
        return epub_path

    except Exception as e:
        log_error(e)
        raise

def test_extract_text_blocks():
    """Test extraction of text and HTML blocks"""
    # Create test EPUB
    epub_path = create_test_epub()
    
    # Create TOC structure
    toc  = [
    TOCEntry(
        title="I: EXPOSITION OF THE TASK OF A PREPARATORY ANALYSIS OF DASEIN",
        href="chap1.xhtml#filepos242677",
        level=1,
        text_blocks=[],
        html_blocks=[]
    ),
    TOCEntry(
        title="9. The Theme of the Analytic of Dasein",
        href="chap1.xhtml#filepos243015", 
        level=2,
        text_blocks=[],
        html_blocks=[]
    ),
    TOCEntry(
        title="10. How the Analytic of Dasein is to be Distinguished from Anthropology, Psychology, and Biology",
        href="chap1.xhtml#filepos264273",
        level=2,
        text_blocks=[],
        html_blocks=[]
    ),
    TOCEntry(
        title="11. The Existential Analytic and the Interpretation of Primitive Dasein. The Difficulties of Achieving a 'Natural Conception of the World'",
        href="chap1.xhtml#filepos288635",
        level=2,
        text_blocks=[],
        html_blocks=[]
    )
]
    
    # Initialize extractor
    extractor = TOCExtractor(str(epub_path))
    extractor.toc_structure = toc
    
    # Extract blocks
    extractor.extract_text_blocks()
    
    # Verify first entry's content
    entry1 = toc[0]
    assert len(entry1.text_blocks) == 0, f"First entry, {entry1.title}, should not have text blocks: {entry1.text_blocks}"
    assert len(entry1.html_blocks) == 0, f"First entry, {entry1.title}, should not have HTML blocks: {entry1.html_blocks}"

    # Verify second entry's content
    entry2 = toc[1]
    assert len(entry2.text_blocks) > 0, f"Second entry, {entry2.title}, should have text blocks: {entry2.text_blocks}"
    assert len(entry2.html_blocks) > 0, f"Second entry, {entry2.title}, should have HTML blocks: {entry2.html_blocks}"
    assert len(entry2.text_blocks) == len(entry2.html_blocks), "Text and HTML block counts do not match"

    # Verify final entry's content
    entry4 = toc[3]
    assert len(entry4.text_blocks) > 0, f"Final entry, {entry4.title}, should have text blocks: {entry4.text_blocks}"
    assert len(entry4.html_blocks) > 0, f"Final entry, {entry4.title}, should have HTML blocks: {entry4.html_blocks}"
    assert len(entry4.text_blocks) == len(entry4.html_blocks), "Text and HTML block counts do not match"



    # Check text/HTML mapping
    for text, html in zip(entry2.text_blocks, entry2.html_blocks):
        # Convert HTML to text for comparison
        html_text = BeautifulSoup(html, 'html.parser').get_text().strip()
        assert text.strip() == html_text, f"Text and HTML blocks do not match: {text} != {html_text}"
        
    # Verify specific content
    expected_text = "WE are ourselves the entities to be analysed."
    assert any(expected_text in block for block in entry2.text_blocks), f"Expected text not found."
    
    # Check second entry
    entry2 = toc[1]
    assert len(entry2.text_blocks) > 0
    assert len(entry2.html_blocks) > 0
    
    # Verify merged blocks
    for text, html in zip(entry2.text_blocks, entry2.html_blocks):
        soup = BeautifulSoup(html, 'html.parser')
        assert text.strip() == soup.get_text().strip()
        
    # Cleanup
    epub_path.unlink()

def test_artifact_filtering():
    """Test filtering of artifacts like page numbers"""
    epub_path = create_test_epub()
    extractor = TOCExtractor(str(epub_path))
    
    # Create simple TOC entry
    toc = [TOCEntry(
        title="Test Chapter",
        href="chap1.xhtml",
        level=1,
        text_blocks=[],
        html_blocks=[]
    )]
    extractor.toc_structure = toc
    
    # Add artifact pattern
    extractor._add_artifact_pattern(r'^H\. \d+$')  # Match page numbers like "H. 42"
    
    # Extract blocks
    extractor.extract_text_blocks(remove_artifacts=True)
    
    # Verify artifacts filtered
    assert not any("H. 42" in block for block in toc[0].text_blocks)
    
    # Cleanup
    epub_path.unlink()

def test_block_merging():
    """Test merging of incomplete sentences"""
    epub_path = create_test_epub()
    extractor = TOCExtractor(str(epub_path))
    
    # Create test blocks
    text_blocks = [
        "This is an incomplete",
        "sentence that should merge.",
        "This is a complete sentence.",
        "Another incomplete",
        "sentence to merge."
    ]
    
    html_blocks = [
        "<p>This is an incomplete</p>",
        "<p>sentence that should merge.</p>",
        "<p>This is a complete sentence.</p>",
        "<p>Another incomplete</p>",
        "<p>sentence to merge.</p>"
    ]
    
    merged_text, merged_html = extractor._merge_blocks(text_blocks, html_blocks)
    
    assert len(merged_text) == 3
    assert "This is an incomplete sentence that should merge." in merged_text
    assert "This is a complete sentence." in merged_text
    assert "Another incomplete sentence to merge." in merged_text
    
    # Cleanup
    epub_path.unlink()



if __name__ == '__main__':
    pytest.main([__file__])