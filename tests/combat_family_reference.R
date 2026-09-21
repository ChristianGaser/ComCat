# combat_family_reference.R — reference outputs from the R ComBatFamily package
#
# Called by tests/test_combat_family.py:
#     Rscript combat_family_reference.R <dir>
# <dir> must contain data.csv (n x p, no header), batch.txt, covar.csv (n x k,
# no header).  Harmonized data are written to <dir>/r_<method>.csv.
#
# Requires: remotes::install_github("andy1764/ComBatFamily")  (pulls in gamlss)

suppressPackageStartupMessages({
  library(ComBatFamily)
  library(gamlss)
})

dir <- commandArgs(trailingOnly = TRUE)[1]
dat <- as.matrix(read.csv(file.path(dir, "data.csv"), header = FALSE))
bat <- as.factor(readLines(file.path(dir, "batch.txt")))
covar <- read.csv(file.path(dir, "covar.csv"), header = FALSE)
names(covar) <- paste0("x", seq_len(ncol(covar)))
f_mu <- as.formula(paste("y ~", paste(names(covar), collapse = " + ")))
f_sigma <- as.formula(paste("~", paste(names(covar), collapse = " + ")))

out <- function(x, name) {
  write.table(x, file.path(dir, paste0("r_", name, ".csv")), sep = ",",
              row.names = FALSE, col.names = FALSE)
}

out(combat(dat, bat, covar, f_mu)$dat.combat, "combat_eb")
out(combat(dat, bat, covar, f_mu, eb = FALSE)$dat.combat, "combat_noeb")

cov95 <- covfam(dat, bat, covar, lm, f_mu)
out(cov95$dat.covbat, "covbat")
writeLines(as.character(cov95$n.pc), file.path(dir, "r_covbat_npc.txt"))
out(covfam(dat, bat, covar, lm, f_mu, n.pc = 3)$dat.covbat, "covbat_npc3")

# Tight convergence so the comparison is not limited by gamlss' default
# stopping rule (c.crit = 0.001)
ctrl <- gamlss.control(c.crit = 1e-10, n.cyc = 500, trace = FALSE)
ictrl <- glim.control(cc = 1e-10, cyc = 500)
out(combatls(dat, bat, covar, f_mu, sigma.formula = f_sigma,
             control = ctrl, i.control = ictrl)$dat.combat, "combatls")
