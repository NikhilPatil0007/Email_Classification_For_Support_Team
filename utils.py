import spacy
import re
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define regex patterns for specific PII fields
FULL_NAME_PATTERN = r'My name is ([A-Za-z\s\.\-]+)[\.,]|Name\s*:\s*([A-Za-z\s\.\-]+)[\.,]'
EMAIL_PATTERN = r'You can reach me at ([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})|Email\s*:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
PHONE_PATTERN = r'My [Cc]ontact number is\s*([+\d\s\-\(\)\.]+)|Phone\s*:\s*([+\d\s\-\(\)\.]+)|(\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9})'
DOB_PATTERN = r'[Dd]ate of [Bb]irth\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})'
AADHAR_PATTERN = r'[Aa]adhar(?:\s*[Cc]ard)?\s*(?:[Nn]umber)?\s*:?\s*(\d{4}\s*\d{4}\s*\d{4}|\d{12})'
CREDIT_DEBIT_PATTERN = r'[Cc](?:redit|ard)\s*(?:[Nn]umber)?\s*:?\s*(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{16})'
CVV_PATTERN = r'[Cc][Vv][Vv]\s*(?:[Nn]umber)?\s*:?\s*(\d{3,4})'
EXPIRY_PATTERN = r'[Ee]xpir(?:y|ation)\s*[Dd]ate\s*:?\s*(\d{1,2}[-/\.]\d{2,4}|\d{2}[-/\.]\d{2})'

def preprocess_text(text):
    """Clean and normalize text before processing. So that It impoves Model Consistency.
    Args :
        text: Email Text to Preprocess before Masking.
    """
    # Handle multiple consecutive newlines
    text = re.sub(r'\n{2,}', '\n', text)
    # Replace all remaining newlines with space
    text = text.replace('\n', ' ')
    # Remove excessive whitespaces So It can reduce Number of tokens
    text = re.sub(r'\s+', ' ', text)
    # Replace common unicode whitespace variants
    text = text.replace('\xa0', ' ')
    # Remove quotes that might be present in pasted emails
    text = text.replace('"', '')

    return text.strip()

def extract_entities(email_text):
    """
    Extract PII entities from text using regex patterns and spaCy

    Args:
        email_text (str): The email text to extract entities from

    Returns:
        list: List of dictionaries containing entity information
    """
    if not email_text or len(email_text.strip()) == 0:
        return []

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully loaded Spacy model")
    except OSError as e:
        logger.error(f"Error loading Spacy model: {e}")
        raise

    preprocessed_text = preprocess_text(email_text)
    entities = []

    #Extract full names and also Postitons of Entity
    name_matches = re.finditer(FULL_NAME_PATTERN, preprocessed_text)
    for match in name_matches:
        name = next((g for g in match.groups() if g), "")
        if name:
            start_idx = email_text.find(name)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(name),
                    "text": name,
                    "type": "full_name"
                })

    #Extract email addresses and also Postitons of Entity
    email_matches = re.finditer(EMAIL_PATTERN, preprocessed_text)
    for match in email_matches:
        email = next((g for g in match.groups() if g), "")
        if email:
            start_idx = email_text.find(email)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(email),
                    "text": email,
                    "type": "email"
                })

    #Extract phone numbers and also Postitons of Entity
    phone_matches = re.finditer(PHONE_PATTERN, preprocessed_text)
    for match in phone_matches:
        phone = next((g for g in match.groups() if g), "")
        if phone:
            start_idx = email_text.find(phone)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(phone),
                    "text": phone,
                    "type": "phone_number"
                })

    #Extract date of birth and also Postitons of Entity
    dob_matches = re.finditer(DOB_PATTERN, preprocessed_text)
    for match in dob_matches:
        dob = match.group(1)
        if dob:
            start_idx = email_text.find(dob)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(dob),
                    "text": dob,
                    "type": "dob"
                })

    #Extract Aadhar numbers and also Postitons of Entity
    aadhar_matches = re.finditer(AADHAR_PATTERN, preprocessed_text)
    for match in aadhar_matches:
        aadhar = match.group(1)
        if aadhar:
            start_idx = email_text.find(aadhar)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(aadhar),
                    "text": aadhar,
                    "type": "aadhar_num"
                })

    #Extract credit/debit card numbers and also Postitons of Entity
    card_matches = re.finditer(CREDIT_DEBIT_PATTERN, preprocessed_text)
    for match in card_matches:
        card = match.group(1)
        if card:
            start_idx = email_text.find(card)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(card),
                    "text": card,
                    "type": "credit_debit_no"
                })

    #Extract CVV numbers and also Postitons of Entity
    cvv_matches = re.finditer(CVV_PATTERN, preprocessed_text)
    for match in cvv_matches:
        cvv = match.group(1)
        if cvv:
            start_idx = email_text.find(cvv)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(cvv),
                    "text": cvv,
                    "type": "cvv_no"
                })

    #Extract card expiry dates and also Postitons of Entity
    expiry_matches = re.finditer(EXPIRY_PATTERN, preprocessed_text)
    for match in expiry_matches:
        expiry = match.group(1)
        if expiry:
            start_idx = email_text.find(expiry)
            if start_idx != -1:
                entities.append({
                    "start": start_idx,
                    "end": start_idx + len(expiry),
                    "text": expiry,
                    "type": "expiry_no"
                })

    # Use spaCy for additional PII detection
    doc = nlp(preprocessed_text)

    # Used spaCy's more advanced NER capabilities
    # Created a mapping from character indices in preprocessed_text to original email_text
    char_mapping = {}
    preprocessed_idx = 0
    original_idx = 0
    
    while preprocessed_idx < len(preprocessed_text) and original_idx < len(email_text):
        if preprocessed_text[preprocessed_idx] == email_text[original_idx]:
            char_mapping[preprocessed_idx] = original_idx
            preprocessed_idx += 1
            original_idx += 1
        else:
            # Skip characters in original text that were removed during preprocessing
            original_idx += 1
    
    # Process spaCy entities with proper position mapping
    for ent in doc.ents:
        entity_type = None
        
        # Map spaCy entity types to our entity types
        if ent.label_ == "PERSON":
            entity_type = "full_name"
        elif ent.label_ == "DATE":
            # Check if it looks like a date of birth or expiry date
            if re.search(DOB_PATTERN, ent.text):
                entity_type = "dob"
            elif re.search(EXPIRY_PATTERN, ent.text):
                entity_type = "expiry_no"
        elif ent.label_ == "CARDINAL" or ent.label_ == "MONEY":
            # Check if it looks like a credit card, aadhar, or other sensitive numbers
            if re.search(CREDIT_DEBIT_PATTERN, ent.text):
                entity_type = "credit_debit_no"
            elif re.search(AADHAR_PATTERN, ent.text):
                entity_type = "aadhar_num"
            elif re.search(CVV_PATTERN, ent.text):
                entity_type = "cvv_no"
        elif ent.label_ == "ORG" and "@" in ent.text:
            # Sometimes spaCy identifies emails as organizations
            entity_type = "email"
            
        if entity_type:
            # Check if this entity overlaps with any existing entity
            already_captured = False
            ent_start_idx = char_mapping.get(ent.start_char, -1)
            
            if ent_start_idx != -1:
                ent_end_idx = char_mapping.get(min(ent.end_char, len(preprocessed_text)-1), 
                                              ent_start_idx + len(ent.text))
                
                # Check for overlap with existing entities
                for entity in entities:
                    if (entity["start"] <= ent_end_idx and 
                        entity["end"] >= ent_start_idx):
                        already_captured = True
                        break
                
                if not already_captured:
                    # Get the actual text from the original email
                    original_text = email_text[ent_start_idx:ent_end_idx]
                    
                    entities.append({
                        "start": ent_start_idx,
                        "end": ent_end_idx,
                        "text": original_text,
                        "type": entity_type
                    })

    # Sort entities by start position
    entities.sort(key=lambda x: x["start"])

    return entities

def mask_pii(email_text):
    """
    Mask PII in email text and return both masked text and entity list.
    
    Args:
        email_text (str): Original email text
        
    Returns:
        tuple: (masked_text, list_of_masked_entities)
            - masked_text: Text with PII replaced by entity type tags
            - list_of_masked_entities: List of dictionaries with entity information
    """
    if not email_text or len(email_text.strip()) == 0:
        return "", []
    
    # Extract all entities from text
    entities = extract_entities(email_text)
    
    # Create a copy of the original text
    masked_text = email_text
    
    # Process entities in reverse order to avoid index shifting when replacing
    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
        start, end = entity["start"], entity["end"]
        entity_type = entity["type"]
        original_text = entity["text"]
        
        # Replace the entity with a tag
        masked_text = masked_text[:start] + f"<{entity_type}>" + masked_text[end:]
    
    # Format entities for output 
    formatted_entities = []
    for entity in entities:
        formatted_entities.append({
            "position": [entity["start"], entity["end"]],
            "classification": entity["type"],
            "entity": entity["text"]
        })
    
    return masked_text, formatted_entities

