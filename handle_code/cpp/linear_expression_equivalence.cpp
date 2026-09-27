/*
中文说明：判断两组只含大写变量、加法和减法的线性表达式定义是否等价。
解题方法：把每个表达式节点递归展开成 26 维系数向量；叶子变量的对应维为 1，
加减节点按符号合并两个子向量。分别求出两组定义的根向量后直接比较。
假设定义无环，且每组第一行左值是根。复杂度 O(26·定义数)，空间 O(26·定义数)。

English: Check whether two systems of uppercase-variable linear expressions
using + and - are equivalent. Recursively expand each node into a 26-component
coefficient vector and compare the two root vectors. Definitions are assumed
acyclic and the first left-hand variable is the root. O(26*n) time and space.
*/
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
