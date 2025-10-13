public with sharing class OrgHealthScanner {

    // ---------- Instance Properties ----------
    public String objectApiName { get; set; }
    public String result { get; set; }

    // ---------- Constructor ----------
    public OrgHealthScanner(String objName) {
        this.objectApiName = objName;
        this.result = '';
    }

    // ---------- Run full scan ----------
    public void runFullScan() {
        String summary = '';

        summary += '\n=== FORMULA FIELD ANALYSIS ===\n';
        summary += analyzeFormulaFields(objectApiName);

        summary += '\n\n=== FIELD FILL RATE ANALYSIS ===\n';
        summary += calculateFillRates(objectApiName);

        summary += '\n\n=== APEX REFERENCE SCAN (OBJECT) ===\n';
        summary += findLocalApexReferences(objectApiName);

        this.result = summary;
    }

    // ---------- Formula Fields ----------
    private String analyzeFormulaFields(String objName) {
        List<String> lines = new List<String>();
        Integer count = 0;
        Schema.SObjectType sObj = Schema.getGlobalDescribe().get(objName);
        if (sObj == null) return 'Error: Object not found: ' + objName;

        for (Schema.SObjectField f : sObj.getDescribe().fields.getMap().values()) {
            if (f.getDescribe().isCalculated()) {
                count++;
                lines.add('- ' + f.getDescribe().getName());
            }
        }
        if (count == 0) return 'No formula fields found.';
        return 'Found ' + count + ' formula fields:\n' + String.join(lines, '\n');
    }

    // ---------- Fill Rate ----------
    private String calculateFillRates(String objName) {
        List<String> output = new List<String>();
        Schema.SObjectType sObj = Schema.getGlobalDescribe().get(objName);
        if (sObj == null) return 'Invalid object: ' + objName;
        
        Map<String, Schema.SObjectField> fields = sObj.getDescribe().fields.getMap();
        Integer totalRecords;
        
        try {
            totalRecords = Database.countQuery('SELECT COUNT() FROM ' + objName);
        } catch (Exception e) {
            return 'Unable to count records for ' + objName + ': ' + e.getMessage();
        }
        
        if (totalRecords == 0) return 'No records found for ' + objName + '.';
        
        for (String f : fields.keySet()) {
            Schema.DescribeFieldResult dfr = fields.get(f).getDescribe();
            if (!dfr.isCustom() || dfr.isCalculated() || dfr.isAutoNumber()) continue;
            
            try {
                Integer filled = Database.countQuery('SELECT COUNT() FROM ' + objName + ' WHERE ' + f + ' != null');
                Decimal rate = (Decimal)filled / totalRecords * 100;
                String line = f + ': ' + rate.setScale(2) + '%';
                if (rate < 5) line += ' ⚠️ Very low fill rate';
                output.add(line);
            } catch (Exception e) {
                // skip invalid field types
            }
        }
        
        if (output.isEmpty()) return 'No custom fields analyzed for ' + objName + '.';
        return 'Analyzed ' + output.size() + ' custom fields.\n' + String.join(output, '\n');
    }

    // ---------- Apex references for object ----------
    private String findLocalApexReferences(String objName) {
        List<String> matches = new List<String>();
        String keyword = objName.toLowerCase();

        // Classes
        for (ApexClass cls : [SELECT Name, Body FROM ApexClass]) {
            if (cls.Body != null && cls.Body.toLowerCase().contains(keyword)) {
                matches.add('Class: ' + cls.Name);
            }
        }

        // Triggers
        for (ApexTrigger trg : [SELECT Name, Body FROM ApexTrigger]) {
            if (trg.Body != null && trg.Body.toLowerCase().contains(keyword)) {
                matches.add('Trigger: ' + trg.Name);
            }
        }

        if (matches.isEmpty()) return 'No Apex references found for ' + objName;
        return 'Found ' + matches.size() + ' Apex references:\n' + String.join(matches, '\n');
    }

    // ---------- Apex references for a specific field (with line numbers) ----------
    public static Map<String, List<String>> findFieldApexReferences(String objectApi, String fieldApi) {
        Map<String, List<String>> results = new Map<String, List<String>>();
        String token = fieldApi.toLowerCase();

        List<String> references = new List<String>();

        // Classes
        for (ApexClass cls : [SELECT Name, Body FROM ApexClass]) {
            if (cls.Body != null) {
                Integer lineNum = 1;
                for (String line : cls.Body.split('\n')) {
                    if (line.toLowerCase().contains(token)) {
                        references.add('Class: ' + cls.Name + ' → Line ' + lineNum + ': ' + line.trim());
                    }
                    lineNum++;
                }
            }
        }

        // Triggers
        for (ApexTrigger trg : [SELECT Name, Body FROM ApexTrigger]) {
            if (trg.Body != null) {
                Integer lineNum = 1;
                for (String line : trg.Body.split('\n')) {
                    if (line.toLowerCase().contains(token)) {
                        references.add('Trigger: ' + trg.Name + ' → Line ' + lineNum + ': ' + line.trim());
                    }
                    lineNum++;
                }
            }
        }

        if (references.isEmpty()) {
            references.add('No Apex references found for ' + objectApi + '.' + fieldApi);
        }

        results.put(objectApi + '.' + fieldApi, references);
        return results;
    }

    // ---------- Static helper for Anonymous Testing ----------
    public static void runTestScan(String objName) {
        OrgHealthScanner s = new OrgHealthScanner(objName);
        s.runFullScan();
        System.debug(s.result);
    }

    public static void runFieldTest(String objName, String fieldName) {
        Map<String, List<String>> res = findFieldApexReferences(objName, fieldName);
        System.debug('Field Usage: ' + res);
    }
}