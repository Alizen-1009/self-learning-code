#include<bits/stdc++.h>
#define ll long long
using namespace std;

const int N = 2e5 + 7;
int a[N], p[N];
bool is_special[N];
int n, k;

void solve() {
    cin >> n >> k;
    for (int i = 1; i <= n; i++) cin >> a[i];

    fill(is_special + 1, is_special + n + 1, false);
    for (int i = 1; i <= k; i++) {
        cin >> p[i];
        is_special[p[i]] = true;
    }

    int x = a[p[1]];
    for (int i = 1; i <= n; i++) a[i] ^= x;

    int total_blocks = 0;
    int max_blocks = 0;
    int cur_blocks = 0;

    for (int i = 1; i <= n; i++) {
        if (is_special[i]) {
            total_blocks += cur_blocks;
            max_blocks = max(max_blocks, cur_blocks);
            cur_blocks = 0;
            continue;
        }
        if (a[i] == 1 && (i == 1 || is_special[i - 1] || a[i - 1] == 0)) {
            cur_blocks++;
        }
    }

    total_blocks += cur_blocks;
    max_blocks = max(max_blocks, cur_blocks);

    cout << max(total_blocks, 2 * max_blocks) << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}