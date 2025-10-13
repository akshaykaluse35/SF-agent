public with sharing class OrgAnalyzer {

    public class AnalyzerRequest {
        @InvocableVariable(label='Scan Type') public String scanType;
        // Values: 'AbandonedFiles'
    }

    public class AnalyzerResult {
        @InvocableVariable(label='Type') public String type;
        @InvocableVariable(label='Label') public String label;
        @InvocableVariable(label='Value') public String value;
    }

    @InvocableMethod(label='Run Org Analyzer')
    public static List<AnalyzerResult> runAnalyzer(List<AnalyzerRequest> requests){
        List<AnalyzerResult> results = new List<AnalyzerResult>();
        if(requests == null || requests.isEmpty()) return results;

        for(AnalyzerRequest req : requests){
            if(req.scanType == 'AbandonedFiles'){
                results.addAll(getAbandonedFiles());
            }
        }
        return results;
    }

    private static List<AnalyzerResult> getAbandonedFiles(){
        List<AnalyzerResult> res = new List<AnalyzerResult>();

        List<ContentDocument> docs = [
            SELECT Id, Title, CreatedDate, (SELECT Id FROM ContentDocumentLinks)
            FROM ContentDocument
            ORDER BY CreatedDate ASC
            LIMIT 200
        ];

        for(ContentDocument cd : docs){
            if(cd.ContentDocumentLinks.isEmpty()){
                AnalyzerResult r = new AnalyzerResult();
                r.type = 'AbandonedFile';
                r.label = cd.Title;
                r.value = 'FileId:' + cd.Id + '|Created:' + cd.CreatedDate;
                res.add(r);
            }
        }
        return res;
    }
}