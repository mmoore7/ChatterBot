from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

class IndexedTextSearch:
    """
    :param statement_comparison_function: A comparison class.
        Defaults to ``LevenshteinDistance``.

    :param search_page_size:
        The maximum number of records to load into memory at a time when searching.
        Defaults to 1000
    """

    name = 'indexed_text_search'

    def __init__(self, chatbot, **kwargs):
        from chatterbot.comparisons import LevenshteinDistance

        self.chatbot = chatbot

        statement_comparison_function = kwargs.get(
            'statement_comparison_function',
            LevenshteinDistance
        )

        self.compare_statements = statement_comparison_function(
            language=self.chatbot.storage.tagger.language
        )

        self.search_page_size = kwargs.get(
            'search_page_size', 1000
        )

        self.alt_stop_words = [
            "registry",
            "reigistry",
            "reigstry",
            "stores",
            "master",
            "file",
            "masterfile",
            "INI",
            "ini",
            "stored"
        ]
        self.stop_words = stopwords.words('english')
        self.stop_words.remove('ain')
        self.stop_words.extend(["'s", 'whats', "what", "which", "that", "?", ".", ","])
        self.st = LancasterStemmer()

    def search(self, input_statement, **additional_parameters):
        """
        Search for close matches to the input. Confidence scores for
        subsequent results will order of increasing value.

        :param input_statement: A statement.
        :type input_statement: chatterbot.conversation.Statement

        :param **additional_parameters: Additional parameters to be passed
            to the ``filter`` method of the storage adapter when searching.

        :rtype: Generator yielding one closest matching statement at a time.
        """
        self.chatbot.logger.info('Beginning search for close text match')

        # Check if the question is related to master files or registries
        if any(word in self.alt_stop_words for word in word_tokenize(input_statement.text.lower())):
            self.chatbot.logger.info(f"Alternative stop words found for '{input_statement.text}'")
            input_statement_tokenized = [x for x in word_tokenize(input_statement.text.lower()) if x not in self.stop_words]
            input_search_lst = [f"NOUN:{self.st.stem(token)}" for token in input_statement_tokenized]
            input_search_text = " ".join(input_search_lst)
        else:
            self.chatbot.logger.info("Alternative stop words not found. Using default search text")
            input_search_text = input_statement.search_text

        if not input_statement.search_text:
            self.chatbot.logger.warn(
                'No value for search_text was available on the provided input'
            )

            input_search_text = self.chatbot.storage.tagger.get_text_index_string(
                input_statement.text
            )

        search_parameters = {
            'search_text_contains': input_search_text,
            'persona_not_startswith': 'bot:',
            'page_size': self.search_page_size
        }

        if additional_parameters:
            search_parameters.update(additional_parameters)

        self.chatbot.logger.info(f"Search function is using '{input_search_text}' for filter")
        statement_list = self.chatbot.storage.filter(**search_parameters)

        best_confidence_so_far = 0

        self.chatbot.logger.info('Processing search results, cross your fingers')
        for statement in statement_list:
            # self.chatbot.logger.info(f"Found statement: {statement}")

            confidence = self.compare_statements(input_statement, statement)
            if confidence > best_confidence_so_far:
                best_confidence_so_far = confidence
                statement.confidence = confidence

                self.chatbot.logger.info('Similar text found: {} {}'.format(
                    statement.text, confidence
                ))

                yield statement

class TextSearch:
    """
    :param statement_comparison_function: A comparison class.
        Defaults to ``LevenshteinDistance``.

    :param search_page_size:
        The maximum number of records to load into memory at a time when searching.
        Defaults to 1000
    """

    name = 'text_search'

    def __init__(self, chatbot, **kwargs):
        from chatterbot.comparisons import LevenshteinDistance

        self.chatbot = chatbot

        statement_comparison_function = kwargs.get(
            'statement_comparison_function',
            LevenshteinDistance
        )

        self.compare_statements = statement_comparison_function(
            language=self.chatbot.storage.tagger.language
        )

        self.search_page_size = kwargs.get(
            'search_page_size', 1000
        )

    def search(self, input_statement, **additional_parameters):
        """
        Search for close matches to the input. Confidence scores for
        subsequent results will order of increasing value.

        :param input_statement: A statement.
        :type input_statement: chatterbot.conversation.Statement

        :param **additional_parameters: Additional parameters to be passed
            to the ``filter`` method of the storage adapter when searching.

        :rtype: Generator yielding one closest matching statement at a time.
        """
        self.chatbot.logger.info('Beginning search for close text match')

        search_parameters = {
            'persona_not_startswith': 'bot:',
            'page_size': self.search_page_size
        }

        if additional_parameters:
            search_parameters.update(additional_parameters)

        statement_list = self.chatbot.storage.filter(**search_parameters)

        best_confidence_so_far = 0

        self.chatbot.logger.info('Processing search results')

        # Find the closest matching known statement
        for statement in statement_list:
            confidence = self.compare_statements(input_statement, statement)

            if confidence > best_confidence_so_far:
                best_confidence_so_far = confidence
                statement.confidence = confidence

                self.chatbot.logger.info('Similar text found: {} {}'.format(
                    statement.text, confidence
                ))

                yield statement
