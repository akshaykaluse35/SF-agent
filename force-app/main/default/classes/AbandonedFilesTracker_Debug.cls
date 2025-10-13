public with sharing class AbandonedFilesTracker_Debug {

    @InvocableMethod(label='Track Abandoned Files (Debug Only)' 
                     description='Finds and logs files not linked to any record for audit or AI health checks')
    public static void findAbandonedFiles() {
        System.debug('==== [AbandonedFilesTracker_Debug] START ====');

        try {
            // Step 1️⃣: Query a limited batch of ContentDocuments (you can expand this via batch later)
            List<ContentDocument> allDocs = [
                SELECT Id, Title, FileType, CreatedDate, CreatedById, LatestPublishedVersionId
                FROM ContentDocument
                ORDER BY CreatedDate DESC
                LIMIT 500
            ];

            if (allDocs.isEmpty()) {
                System.debug('⚪ No ContentDocuments found in org.');
                return;
            }

            // Step 2️⃣: Collect IDs
            Set<Id> allDocIds = new Set<Id>();
            for (ContentDocument cd : allDocs) {
                allDocIds.add(cd.Id);
            }

            // Step 3️⃣: Get linked documents (filter using IN as Salesforce requires)
            Set<Id> linkedDocIds = new Set<Id>();
            for (ContentDocumentLink link : [
                SELECT ContentDocumentId
                FROM ContentDocumentLink
                WHERE ContentDocumentId IN :allDocIds
            ]) {
                linkedDocIds.add(link.ContentDocumentId);
            }

            // Step 4️⃣: Identify orphaned (abandoned) ones
            List<ContentDocument> orphanedDocs = new List<ContentDocument>();
            for (ContentDocument cd : allDocs) {
                if (!linkedDocIds.contains(cd.Id)) {
                    orphanedDocs.add(cd);
                }
            }

            // Step 5️⃣: Log Results
            if (orphanedDocs.isEmpty()) {
                System.debug('✅ No abandoned files found in this scan.');
            } else {
                System.debug('⚠️ Found ' + orphanedDocs.size() + ' abandoned files:');
                Integer counter = 1;
                for (ContentDocument cd : orphanedDocs) {
                    System.debug(
                        counter + ') 📁 ' + cd.Title +
                        ' | Id: ' + cd.Id +
                        ' | Type: ' + cd.FileType +
                        ' | CreatedBy: ' + cd.CreatedById +
                        ' | CreatedDate: ' + cd.CreatedDate
                    );
                    counter++;
                }
            }

        } catch (Exception e) {
            System.debug('❌ Error while finding abandoned files: ' + e.getMessage());
        }

        System.debug('==== [AbandonedFilesTracker_Debug] END ====');
    }
}