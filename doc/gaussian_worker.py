import psycopg2
import time
import logging
import random
import subprocess
import socket

dbparams = {
# In this example file, the database connection parameters (server, password, etc),
# has been removed. This file is mainly to show an example of how a SQL database
# was used to queue and dispatch Gaussian calculations.
}


from bde.gaussian import GaussianRunner


def run_optimization():

    with psycopg2.connect(**dbparams) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            WITH cte AS (
            SELECT id, smiles, type
            FROM compound
            WHERE status = 'not started'
            ORDER BY id
            LIMIT 1
            FOR UPDATE
            )

            UPDATE compound SET status = 'in progress',
                                queued_at = CURRENT_TIMESTAMP,
                                node = %s
            FROM cte
            WHERE compound.id = cte.id
            RETURNING compound.id, compound.smiles, compound.type;
            """, (socket.gethostname(),))

            cid, smiles, type_ = cur.fetchone()
            
    conn.close()

    try:
        
        runner = GaussianRunner(smiles, cid, type_)
        molstr, enthalpy, freeenergy, scfenergy, log = runner.process()
        
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:   
                cur.execute("""
                UPDATE compound
                SET status = 'finished', 
                    mol = %s, enthalpy = %s, 
                    freeenergy = %s, scfenergy= %s,
                    run_at = CURRENT_TIMESTAMP,
                    logfile = %s
                WHERE id = %s;""", 
                (molstr, enthalpy, freeenergy, scfenergy, log, cid))

        conn.close()


    except Exception as ex:
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE compound
                SET status = 'error', 
                    error = %s, 
                    run_at = CURRENT_TIMESTAMP
                WHERE id = %s;""", (str(ex), cid))  
        
        conn.close()
                
    return cid
                

if __name__ == "__main__":
    
    start_time = time.time()

    # Add a random delay to avoid race conditions at the start of the job
    time.sleep(random.uniform(0, 1*60))
    
    while (time.time() - start_time) < (86400 * 9):  # Time in days
        

        try:
            run_optimization()
            
        except psycopg2.OperationalError:
            time.sleep(5 + random.uniform(0, 60))
