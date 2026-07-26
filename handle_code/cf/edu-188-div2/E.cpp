#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
string buildS(int x, array<int, 10>& need, int& sumTail) {
    string t = "";
    need.fill(0);
    sumTail = 0;

    while (true) {
        string cur = to_string(x);
        t += cur;
        int nx = 0;
        for (char c : cur) {
            int d = c - '0';
            need[d]++;
            sumTail += d;
            nx += d;
        }
        if (x <= 9) break;
        x = nx;
    }
    return t;
}
void solve() {
    string s;
    cin >> s;
    if ((int)s.size() == 1) {
        cout << s << '\n';
        return;
    }

    array<int, 10> cnt{};
    int sum = 0;
    for (auto c : s) {
        sum += c - '0';
        cnt[c - '0']++;
    }

    for (int target = sum; target >= 1; target--) {
        // pf(target);
        array<int, 10> need{};
        int sumTail = 0;
        string tail = buildS(target, need, sumTail);
        if ((int)tail.size() >= (int)s.size()) continue;
        if (target + sumTail != sum) continue;
        // pf(tail);

        bool ok = true;
        for (int d = 0; d <= 9; d++) {
            if (need[d] > cnt[d]) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        array<int, 10> rem = cnt;
        for (int d = 0; d <= 9; d++) rem[d] -= need[d];

        int first = -1;
        for (int d = 1; d <= 9; d++) {
            if (rem[d] > 0) {
                first = d;
                break;
            }
        }

        string xstr = "";
        xstr.push_back(char('0' + first));
        rem[first]--;

        for (int d = 0; d <= 9; d++) {
            xstr.append(rem[d], char('0' + d));
        }
        cout << xstr + tail << "\n";
        return;
    }

}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}