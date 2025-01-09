import dataclasses
import pytest
from bs4 import BeautifulSoup
from epubkit.parser import (
    HTMLCategoryExtractor,
    CategoryPattern, 
    CategoryExtractionError,
    CategoryType,
    PositionRules,
    TagInfo,
    ImmutableTagInfo,
    html_to_selections,
    CategoryDict,
    HTML,
    TagMatcher,
    ConditionalRule
)
from typing import Dict, Set, Tuple, List

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
                position_rules=PositionRules(
                    depth_requirements={'span': 1}
            )
    )})
    
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
                        <span class="calibre12" id="span1">
                            <span data-custom="value">
                                <span class="calibre13">Content</span>
                            </span>
                        </span>
                    </p>
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
            position_rules=PositionRules(
                depth_requirements={'span': 1}
            )
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
                    classes=frozenset(['italic'])
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
        <p class="calibre_8">Title</p>
        <div class="calibre_9"> </div>
        <p class="calibre_6">Content</p>
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
