/**
 * @description Retrieves performance data for both Dashboards and Reports.
 * Note: This class uses the 'LastViewedDate' to find recently used items.
 * The 'viewCount' is SIMULATED. For true view counts, use Event Monitoring.
 */
public with sharing class TopDashboardsSummaryReal {

    /**
     * @description Generic wrapper class for an analytics item (Report or Dashboard).
     * Implements Comparable to allow for sorting based on view count.
     */
    public class AnalyticsItemInfo implements Comparable {
        @AuraEnabled public String itemName { get; set; }
        @AuraEnabled public Id itemId { get; set; }
        @AuraEnabled public String itemType { get; set; } // 'Report' or 'Dashboard'
        @AuraEnabled public Integer viewCount { get; set; }
        @AuraEnabled public String folderName { get; set; }

        // Constructor for Dashboards
        public AnalyticsItemInfo(Dashboard d) {
            this.itemName = d.Title;
            this.itemId = d.Id;
            this.itemType = 'Dashboard';
            this.folderName = d.FolderName;
            this.viewCount = generateRandomViewCount();
        }

        // Overloaded constructor for Reports
        public AnalyticsItemInfo(Report r) {
            this.itemName = r.Name;
            this.itemId = r.Id;
            this.itemType = 'Report';
            this.folderName = r.FolderName;
            this.viewCount = generateRandomViewCount();
        }

        private Integer generateRandomViewCount() {
            return Integer.valueOf(Math.random() * 15) + 1;
        }

        // Sorts the list from highest to lowest view count
        public Integer compareTo(Object compareTo) {
            AnalyticsItemInfo compareToInfo = (AnalyticsItemInfo)compareTo;
            if (this.viewCount < compareToInfo.viewCount) return 1;
            if (this.viewCount > compareToInfo.viewCount) return -1;
            return 0;
        }
    }

    /**
     * @description Queries for top-performing reports and dashboards.
     * @param daysAgo The number of days back to look for viewed items.
     * @param folderNameFilter Optional: The exact name of the folder to filter by.
     * @return A sorted, combined list of AnalyticsItemInfo objects.
     */
    @AuraEnabled(cacheable=true)
    public static List<AnalyticsItemInfo> getTopAnalyticsSummary(Integer daysAgo, String folderNameFilter) {
        if (daysAgo == null || daysAgo <= 0) {
            daysAgo = 5;
        }

        List<AnalyticsItemInfo> results = new List<AnalyticsItemInfo>();
        Datetime dateFilter = System.now().addDays(-daysAgo);

        try {
            // --- Query for Dashboards ---
            String dashboardQuery = 'SELECT Id, Title, FolderName, LastViewedDate FROM Dashboard WHERE LastViewedDate >= :dateFilter ';
            if (String.isNotBlank(folderNameFilter)) {
                dashboardQuery += 'AND FolderName = :folderNameFilter ';
            }
            dashboardQuery += 'ORDER BY LastViewedDate DESC LIMIT 50';
            
            for (Dashboard d : Database.query(dashboardQuery)) {
                results.add(new AnalyticsItemInfo(d));
            }

            // --- Query for Reports ---
            String reportQuery = 'SELECT Id, Name, FolderName, LastViewedDate FROM Report WHERE LastViewedDate >= :dateFilter ';
            if (String.isNotBlank(folderNameFilter)) {
                reportQuery += 'AND FolderName = :folderNameFilter ';
            }
            reportQuery += 'ORDER BY LastViewedDate DESC LIMIT 50';

            for (Report r : Database.query(reportQuery)) {
                results.add(new AnalyticsItemInfo(r));
            }

            // Sort the combined list by the simulated view count
            results.sort();

        } catch (Exception e) {
            System.debug('Error querying analytics performance: ' + e.getMessage());
            throw new AuraHandledException('Failed to retrieve analytics data. ' + e.getMessage());
        }

        return results;
    }
}