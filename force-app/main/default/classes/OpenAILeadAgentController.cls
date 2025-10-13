public with sharing class OpenAILeadAgentController {

    // Inner classes for the Gemini API JSON structure
    private class GeminiRequest {
        public List<Content> contents;
        GeminiRequest(List<Content> contents) { this.contents = contents; }
    }
    private class Content {
        public List<Part> parts;
        Content(List<Part> parts) { this.parts = parts; }
    }
    private class Part {
        public String text;
        Part(String text) { this.text = text; }
    }
    private class GeminiResponse {
        public List<Candidate> candidates;
    }
    private class Candidate {
        public Content content;
    }

    @AuraEnabled(cacheable=false)
    public static String getOpenAIResponse(String userMessage, String historyJson) {
        if (String.isBlank(userMessage)) { return 'Please ask a question.'; }

        try {
            // Step 1: Get the Gemini key from your Custom Setting field.
            API_Keys__c settings = API_Keys__c.getOrgDefaults();
            String apiKey = settings.OpenAI_API_Key__c;
            if (String.isBlank(apiKey)) {
                return 'ERROR: API Key is not set in Custom Settings.';
            }

            // Step 2: Fetch leads from Salesforce.
            List<Lead> openLeads = [
                SELECT Name, Company, Title, Industry, LeadSource
                FROM Lead WHERE Status = 'Open - Not Contacted' AND IsConverted = false
                ORDER BY CreatedDate DESC LIMIT 15
            ];

            // Step 3: Construct the request payload for the Gemini API.
            String leadDataForContext = JSON.serialize(openLeads);
            String fullUserMessage = 'You are a Salesforce sales assistant. You have to give sales insgihts to user, greet sales user. Answer to there questions. When asked for top leads, provide a bulleted list using "•". For each lead, provide their name and a BOLDED one-sentence reason. Format your response using HTML (<b> and <br/>).\n\n' +
                                     'User question: ' + userMessage + '\n\n' +
                                     'Here is the current list of open leads to analyze:\n' + leadDataForContext;

            List<Part> parts = new List<Part>{ new Part(fullUserMessage) };
            List<Content> contents = new List<Content>{ new Content(parts) };
            String requestBody = JSON.serialize(new GeminiRequest(contents));

            // Step 4: Perform the callout with the correct model name from your list.
            HttpRequest req = new HttpRequest();
            String endpoint = 'callout:OpenAI_API/v1/models/gemini-2.5-flash:generateContent?key=' + apiKey;
            req.setEndpoint(endpoint);
            req.setMethod('POST');
            req.setHeader('Content-Type', 'application/json;charset=UTF-8');
            req.setBody(requestBody);
            req.setTimeout(60000);

            HttpResponse res = new Http().send(req);

            // Step 5: Process the Gemini-specific response.
            if (res.getStatusCode() == 200) {
                GeminiResponse responseWrapper = (GeminiResponse)JSON.deserialize(res.getBody(), GeminiResponse.class);
                if (responseWrapper != null && !responseWrapper.candidates.isEmpty()) {
                    return responseWrapper.candidates[0].content.parts[0].text;
                }
                return 'The AI returned an empty response.';
            } else {
                return 'Error from Gemini API: ' + res.getStatusCode() + ' - ' + res.getBody();
            }
        } catch (Exception e) {
            throw new AuraHandledException('An Apex error occurred: ' + e.getMessage());
        }
    }
}