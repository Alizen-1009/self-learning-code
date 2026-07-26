#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;

namespace Expression {
    using Vec = array<int, 26>;
    struct node {
        char a, b;
        int flg;
    };

    unordered_map<char, node> mp;
    unordered_map<char, Vec> val;

    void dfs(char x) {
        if (val.count(x)) return;
        if (!mp.count(x)) {
            Vec tmp{};
            tmp[x - 'A'] = 1;
            val[x] = tmp;
            return;
        }
        int a = mp[x].a;
        int b = mp[x].b;
        dfs(a);dfs(b);
        Vec tmp{};
        for (int i = 0; i < 26; i++) {
            tmp[i] = val[a][i] + mp[x].flg * val[b][i];
        }
        val[x] = tmp;
    }

    Vec read(int n) {
        mp.clear();
        val.clear();
        char root = 0;

        for (int i = 1; i <= n; i++) {
            string s;
            cin >> s;
            char c = s[0], a = s[2], op = s[3], b = s[4];
            int flg = 1;
            if (op == '-') flg = -1;
            if (i == 1) root = c;
            mp[c] = { a, b, flg };
        }

        dfs(root);
        return val[root];
    }
}
void solve() {
    int n1, n2;
    while (cin >> n1) {
        auto v1 = Expression::read(n1);
        cin >> n2;
        auto v2 = Expression::read(n2);
        cout << (v1 == v2 ? "YES" : "NO") << '\n';
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
