#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
void solve() {
    unordered_set<int> s;
    int n;
    cin >> n;
    string str;
    cin >> str;
    int pos = 1;
    s.insert(1);
    while (n--) {
        if (str[pos - 1] == 'R') pos++;
        else pos--;
        s.insert(pos);
    }
    cout << (int)s.size() << '\n';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}