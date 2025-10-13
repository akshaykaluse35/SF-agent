public with sharing class UserSecurityScanner {

    public class UserPermissionResult {
        @InvocableVariable public String userName;
        @InvocableVariable public String userEmail;
        @InvocableVariable public Integer permissionSetCount;
    }

    public class UserLoginResult {
        @InvocableVariable public String userName;
        @InvocableVariable public String loginIp;
        @InvocableVariable public Datetime loginTime;
    }

    // ---------- Get top users by permission sets ----------
    public static List<UserPermissionResult> getTopPermissionSetUsers(Integer topN) {
        Map<Id, Integer> userPermCount = new Map<Id, Integer>();

        // Count permission sets per user
        for (AggregateResult ar : [
            SELECT AssigneeId Id, COUNT(PermissionSetId) cnt
            FROM PermissionSetAssignment
            GROUP BY AssigneeId
        ]) {
            userPermCount.put((Id)ar.get('Id'), (Integer)ar.get('cnt'));
        }

        List<UserPermissionResult> results = new List<UserPermissionResult>();
        if (!userPermCount.isEmpty()) {
            List<Id> userIds = new List<Id>(userPermCount.keySet());
            Map<Id, User> users = new Map<Id, User>([SELECT Id, Name, Email FROM User WHERE Id IN :userIds]);

            for (Id uid : userIds) {
                User u = users.get(uid);
                if (u != null) {
                    UserPermissionResult upr = new UserPermissionResult();
                    upr.userName = u.Name;
                    upr.userEmail = u.Email;
                    upr.permissionSetCount = userPermCount.get(uid);
                    results.add(upr);
                }
            }

            // Manual descending sort
            for (Integer i = 0; i < results.size() - 1; i++) {
                for (Integer j = i + 1; j < results.size(); j++) {
                    if (results[i].permissionSetCount < results[j].permissionSetCount) {
                        UserPermissionResult temp = results[i];
                        results[i] = results[j];
                        results[j] = temp;
                    }
                }
            }
        }

        // Return top N
        List<UserPermissionResult> topResults = new List<UserPermissionResult>();
        Integer limitN = (topN != null && topN < results.size()) ? topN : results.size();
        for (Integer i = 0; i < limitN; i++) {
            topResults.add(results[i]);
        }
        return topResults;
    }

    // ---------- Get login history (last N days) ----------
    public static List<UserLoginResult> getUserLoginHistory(Integer lastNDays) {
        List<UserLoginResult> logins = new List<UserLoginResult>();
        Datetime fromDate = Datetime.now().addDays(-lastNDays);

        // Query LoginHistory object
        List<LoginHistory> loginRecords = [
            SELECT UserId, LoginTime, SourceIp
            FROM LoginHistory
            WHERE LoginTime >= :fromDate
            ORDER BY LoginTime DESC
        ];

        // Map UserId to Name
        Set<Id> userIds = new Set<Id>();
        for (LoginHistory lh : loginRecords) userIds.add(lh.UserId);
        Map<Id, User> users = new Map<Id, User>([SELECT Id, Name FROM User WHERE Id IN :userIds]);

        for (LoginHistory lh : loginRecords) {
            UserLoginResult lr = new UserLoginResult();
            User u = users.get(lh.UserId);
            lr.userName = (u != null) ? u.Name : 'Unknown User';
            lr.loginIp = lh.SourceIp;
            lr.loginTime = lh.LoginTime;
            logins.add(lr);
        }

        return logins;
    }

    // ---------- Test Method ----------
    public static void runTest() {
        System.debug('=== Top Permission Set Users ===');
        List<UserPermissionResult> topUsers = getTopPermissionSetUsers(5);
        for (UserPermissionResult u : topUsers) {
            System.debug(u.userName + ' (' + u.userEmail + ') → Permission Sets: ' + u.permissionSetCount);
        }

        System.debug('=== User Login History (last 7 days) ===');
        List<UserLoginResult> loginHistory = getUserLoginHistory(7);
        for (UserLoginResult l : loginHistory) {
            System.debug(l.userName + ' → IP: ' + l.loginIp + ' → LoginTime: ' + l.loginTime);
        }
    }
}