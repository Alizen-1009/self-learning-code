/*
题目：比较两个加减表达式是否恒等
【题意】每组有若干行形如 X=A+B 或 X=A-B 的定义，未定义的大写字母作为
独立基础变量。每组第一行的左侧字母是待比较的根表达式；判断两组根在任意基础
变量取值下是否相等。代码假设依赖关系无环，输入表达式无空格。
【方法】把表达式展开成长度 26 的系数向量。例如 X=A-B 的 A 系数为 1、B
系数为 -1。DFS 递归求两个孩子的向量，加法相加、减法相减，并缓存结果。
比较两组根的向量即可。时间、空间均 O(26·定义数)。

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
