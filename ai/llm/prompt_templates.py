"""
Prompt templates for legal AI assistant.
Contains system prompts, conversation templates, and task-specific prompts.
"""

from typing import Dict, List, Optional


class LegalPromptTemplates:
    """Collection of prompt templates for legal AI assistant."""
    
    # System prompts
    LEGAL_ASSISTANT_SYSTEM = """You are a professional legal AI assistant for a law firm. Your role is to:

1. Provide accurate, helpful legal information based on the provided context
2. Maintain a professional, courteous tone at all times
3. Always include appropriate legal disclaimers
4. Admit when you don't know something rather than guessing
5. Ask clarifying questions when needed
6. Focus on practical, actionable advice

IMPORTANT DISCLAIMERS:
- You are not providing legal advice but general legal information
- Always recommend consulting with a qualified attorney for specific legal matters
- Laws vary by jurisdiction and change over time
- Each case is unique and requires individual assessment

Remember: You are representing a professional law firm, so maintain the highest standards of accuracy and professionalism."""

    DOCUMENT_ANALYSIS_SYSTEM = """You are a legal document analyst. Your task is to:

1. Carefully review and analyze legal documents
2. Identify key provisions, terms, and potential issues
3. Summarize complex legal language in clear, understandable terms
4. Highlight important dates, deadlines, and obligations
5. Flag any unusual or potentially problematic clauses

When analyzing documents:
- Be thorough but concise
- Use bullet points for clarity
- Highlight critical information
- Note any missing or incomplete information
- Suggest follow-up questions or actions if needed"""

    CONVERSATION_SYSTEM = """You are a helpful legal AI assistant handling phone calls for a law firm. Your goals:

1. Provide excellent customer service
2. Gather relevant information efficiently
3. Determine the caller's needs and appropriate next steps
4. Schedule appointments when appropriate
5. Transfer complex matters to human attorneys

Guidelines:
- Be warm, professional, and patient
- Ask one question at a time
- Confirm important details
- Provide realistic expectations about timelines
- Always end with clear next steps"""

    # RAG-based templates
    RAG_QUERY_TEMPLATE = """Context from legal documents:
{context}

Question: {question}

Based on the provided context, please provide a comprehensive answer. If the context doesn't contain enough information to fully answer the question, clearly state what additional information would be needed.

Remember to:
- Cite relevant sections from the context when possible
- Include appropriate legal disclaimers
- Suggest consulting with an attorney for specific legal advice"""

    RAG_DOCUMENT_SUMMARY_TEMPLATE = """Please provide a summary of this legal document:

{document_text}

Include:
1. Document type and purpose
2. Key parties involved
3. Main terms and conditions
4. Important dates and deadlines
5. Notable provisions or potential concerns

Format as a clear, organized summary that a non-lawyer could understand."""

    # Conversation flow templates
    INTAKE_GREETING = """Hello! I'm the AI assistant for [Law Firm Name]. I'm here to help you with your legal inquiry. 

To better assist you, could you please tell me:
1. What type of legal matter brings you to us today?
2. Is this regarding a new issue or an existing case?"""

    APPOINTMENT_SCHEDULING = """I'd be happy to schedule a consultation for you with one of our attorneys. 

To find the best available time:
1. What is your preferred day of the week?
2. Do you prefer morning or afternoon appointments?
3. Is this matter urgent, or can we schedule within the next 1-2 weeks?

The consultation will typically last 30-60 minutes depending on the complexity of your matter."""

    INFORMATION_GATHERING = """To help our attorney prepare for your consultation, could you provide some additional details:

1. When did this legal issue first arise?
2. Have you taken any action on this matter previously?
3. Do you have any relevant documents or correspondence?
4. Are there any upcoming deadlines we should be aware of?"""

    # Specialized practice area templates
    CONTRACT_REVIEW_PROMPT = """Please review this contract and provide analysis focusing on:

1. Key terms and obligations for each party
2. Payment terms and schedules
3. Termination clauses and conditions
4. Liability and indemnification provisions
5. Dispute resolution mechanisms
6. Any unusual or potentially problematic clauses

Contract text:
{contract_text}

Provide your analysis in a clear, organized format suitable for client review."""

    ESTATE_PLANNING_PROMPT = """Based on the client information provided, please analyze their estate planning needs:

Client Information:
{client_info}

Please address:
1. Recommended estate planning documents
2. Tax considerations and strategies
3. Asset protection opportunities
4. Beneficiary designations and planning
5. Potential issues or complications
6. Next steps and timeline"""

    LITIGATION_ASSESSMENT_PROMPT = """Please analyze this potential litigation matter:

Case Information:
{case_info}

Provide assessment on:
1. Strength of potential claims
2. Likely defenses and counterarguments
3. Estimated timeline and process
4. Potential damages or remedies
5. Settlement considerations
6. Recommended strategy and next steps"""

    @classmethod
    def get_system_prompt(cls, prompt_type: str) -> str:
        """Get system prompt by type."""
        prompts = {
            "legal_assistant": cls.LEGAL_ASSISTANT_SYSTEM,
            "document_analysis": cls.DOCUMENT_ANALYSIS_SYSTEM,
            "conversation": cls.CONVERSATION_SYSTEM
        }
        return prompts.get(prompt_type, cls.LEGAL_ASSISTANT_SYSTEM)

    @classmethod
    def format_rag_query(cls, context: str, question: str) -> str:
        """Format RAG query with context."""
        return cls.RAG_QUERY_TEMPLATE.format(context=context, question=question)

    @classmethod
    def format_document_summary(cls, document_text: str) -> str:
        """Format document summary prompt."""
        return cls.RAG_DOCUMENT_SUMMARY_TEMPLATE.format(document_text=document_text)

    @classmethod
    def format_contract_review(cls, contract_text: str) -> str:
        """Format contract review prompt."""
        return cls.CONTRACT_REVIEW_PROMPT.format(contract_text=contract_text)

    @classmethod
    def format_estate_planning(cls, client_info: str) -> str:
        """Format estate planning prompt."""
        return cls.ESTATE_PLANNING_PROMPT.format(client_info=client_info)

    @classmethod
    def format_litigation_assessment(cls, case_info: str) -> str:
        """Format litigation assessment prompt."""
        return cls.LITIGATION_ASSESSMENT_PROMPT.format(case_info=case_info)


class ConversationFlowTemplates:
    """Templates for managing conversation flows."""
    
    GREETING_RESPONSES = [
        "Hello! Welcome to [Law Firm Name]. How can I assist you with your legal matter today?",
        "Good [morning/afternoon]! I'm here to help you with your legal inquiry. What brings you to us today?",
        "Thank you for calling [Law Firm Name]. I'm our AI assistant, and I'm here to help. What can I assist you with?"
    ]
    
    CLARIFICATION_QUESTIONS = [
        "Could you provide a bit more detail about that?",
        "To better understand your situation, could you tell me more about...",
        "I want to make sure I understand correctly. Are you saying that...",
        "Let me clarify - you mentioned... Could you elaborate on that?"
    ]
    
    ESCALATION_PHRASES = [
        "This matter would benefit from speaking directly with one of our attorneys. Let me arrange that for you.",
        "Based on what you've shared, I think it would be best to schedule a consultation with our legal team.",
        "This is exactly the type of case our attorneys handle regularly. Would you like me to connect you with someone?"
    ]
    
    CLOSING_STATEMENTS = [
        "Is there anything else I can help you with today?",
        "Do you have any other questions before we conclude?",
        "What other information can I provide to assist you?"
    ]


class SpecializedPrompts:
    """Specialized prompts for specific legal tasks."""
    
    CLIENT_INTAKE_CHECKLIST = """Client Intake Information Needed:

Personal Information:
- Full name and contact information
- Address and jurisdiction
- Emergency contact

Matter Details:
- Type of legal issue
- Date issue arose
- Parties involved
- Current status/actions taken
- Urgency level
- Desired outcome

Documentation:
- Relevant documents available
- Previous legal representation
- Correspondence related to matter
- Deadlines or time constraints

Next Steps:
- Consultation scheduling
- Document collection
- Attorney assignment
- Fee structure discussion"""

    DOCUMENT_CHECKLIST_TEMPLATE = """Document Review Checklist:

Required Information:
□ Document type and date
□ Parties involved
□ Key terms and provisions
□ Effective dates and durations
□ Financial terms
□ Obligations and responsibilities
□ Termination conditions
□ Dispute resolution procedures
□ Governing law and jurisdiction
□ Signature requirements

Analysis Points:
□ Completeness of terms
□ Potential risks or issues
□ Missing provisions
□ Recommended modifications
□ Compliance considerations"""

    DEADLINE_TRACKING_TEMPLATE = """Legal Deadline Tracking:

Matter: {matter_name}
Client: {client_name}
Attorney: {attorney_name}

Critical Dates:
- Statute of limitations: {statute_date}
- Filing deadlines: {filing_dates}
- Response deadlines: {response_dates}
- Court dates: {court_dates}
- Discovery deadlines: {discovery_dates}

Action Items:
- Immediate actions required
- Upcoming deadlines (next 30 days)
- Long-term milestones
- Client communication needed"""