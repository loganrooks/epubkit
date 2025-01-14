import dataclasses
from pathlib import Path
import bs4
import pytest
from bs4 import BeautifulSoup
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
from epubkit.utils import debug_print_dict, load_epub_html

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
        
