# Security policy

SeatSense handles camera views of people at work and HR data, so security reports are taken seriously.

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately instead:
[**Report a vulnerability**](https://github.com/devanshsingh0704/seatsense/security/advisories/new)
(Security tab → Advisories → *Report a vulnerability*).

Include what is affected, how to reproduce it, and what an attacker could do with it. You will get an
answer within a week; a confirmed issue is fixed first and disclosed after, with credit if you want it.

## Scope

In scope: the dashboard (sign-in, sessions, camera access, HTTPS and the HTTP redirect, the API),
the labelling tools, and anything in this repository that could expose camera images, reports,
accounts or credentials.

Out of scope: the self-signed certificate warning (by design; see the README), and deployments
that bypass the defaults, for example plain HTTP opened to a network with `SEATSENSE_BIND`.

## Supported versions

Only the latest commit on `main` is supported.
